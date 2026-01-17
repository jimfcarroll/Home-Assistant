import { PlaywrightCrawler, RequestQueue, log } from "crawlee";

// ------------------------------
// Helper: URL origin extraction
// ------------------------------
// This is plain Node.js code. Crawlee does not provide URL policy logic for you;
// instead, it gives you hooks (enqueueLinks / transformRequestFunction) where
// you can enforce your own crawl boundaries. This helper is used in those hooks.
function toOrigin(url) {
  try { return new URL(url).origin; } catch { return null; }
}

// ----------------------------------------------------------------------
// Browser-side extraction logic (NOT Crawlee-specific)
// ----------------------------------------------------------------------
// This function runs *inside Chromium*, not in Node.js.
//
// Key point:
// - Crawlee is NOT responsible for content extraction here.
// - Crawlee's job is to get us a live, rendered page and manage its lifecycle.
// - Once we have `page`, we use Playwright directly to decide what "content"
//   means for our application.
//
// This separation is intentional:
//   Crawlee = crawl correctness & orchestration
//   This function = page semantics & extraction
async function extractFromPage(page, opts) {
  return await page.evaluate((includeLinks) => {
    // Small helper local to the browser context.
    // It normalizes whitespace and guards against nulls.
    const norm = (s) => s?.replace(/\s+/g, " ").trim() || "";

    // These reads are pure DOM access. No Crawlee involvement here.
    const title = norm(document.title);
    const h1 = norm(document.querySelector("h1")?.innerText);

    const headings = Array.from(document.querySelectorAll("h2,h3"))
      .slice(0, 12)
      .map((el) => norm(el.innerText));

    const metaDescription = norm(
      document.querySelector('meta[name="description"]')?.content
    );

    // We mutate the DOM here *only to simplify text extraction*.
    // This is safe because each requestHandler invocation gets its own page.
    document.querySelectorAll(
      "script,style,noscript,svg,canvas,iframe,nav,footer,header,aside"
    ).forEach((el) => el.remove());

    const main =
      document.querySelector("main") ||
      document.querySelector("article") ||
      document.body;

    const text = norm(main?.innerText);

    // Link extraction also happens in the browser context.
    // Crawlee will later decide *what to do* with these links.
    let links = [];
    if (includeLinks) {
      links = Array.from(document.querySelectorAll("a[href]"))
        .map((a) => a.href)
        .filter((h) => h.startsWith("http"))
        .slice(0, 200);
    }

    return { title, h1, headings, metaDescription, text, links };
  }, opts.includeLinks);
}

// ----------------------------------------------------------------------
// Crawl orchestration (this is where Crawlee earns its keep)
// ----------------------------------------------------------------------
// Everything below this point is about *crawl correctness*, not extraction.
//
// Crawlee provides:
// - Request queue + deduplication
// - Controlled concurrency
// - Page / browser lifecycle management
// - Failure isolation per request
// - Graceful shutdown semantics
//
// If you removed Crawlee, you would need to reimplement all of that yourself.
export async function runCrawl(opts) {

  // Create a fresh request queue for THIS run only
  const requestQueue = await RequestQueue.open(`run-${Date.now()}`);

  // Seed start URLs explicitly
  await requestQueue.addRequests(
    opts.startUrls.map((u) => ({
      url: u,
      userData: { depth: 0 },
    }))
  );

  const startOrigin = toOrigin(opts.startUrls[0]);
  const allowedOrigin = opts.sameOriginOnly ? startOrigin : null;

  // In-memory aggregation of results for this run.
  // Crawlee itself does not impose a result format.
  const results = [];

  // Local deduplication guard. Crawlee also deduplicates at the queue level;
  // this is an extra safety net for already-loaded URLs.
  const seen = new Set();

  // This object wires your logic into Crawlee's crawl engine.
  const crawler = new PlaywrightCrawler({
    requestQueue,

    useSessionPool: false,

    // Force Playwright to use the system-installed Chrome provided by the base image.
    // This image does NOT use Playwright-managed browsers (.cache / ms-playwright).
    // Without this, Playwright will fail to launch with a generic "install browsers" error.
    launchContext: {
      launchOptions: {
        executablePath: "/usr/bin/google-chrome",
        args: ["--no-sandbox", "--disable-setuid-sandbox"],
      },
    },

    // Global crawl bounds (enforced by Crawlee)
    maxRequestsPerCrawl: opts.maxPages,
    maxConcurrency: opts.concurrency,
    requestHandlerTimeoutSecs: opts.timeoutSecs,

    // Hook that runs before navigation. Crawlee manages page creation;
    // you get a chance to configure it.
    preNavigationHooks: [
      async ({ page }) => {
        if (opts.userAgent) await page.setUserAgent(opts.userAgent);
      }
    ],

    // This function is called once per queued URL.
    // Crawlee controls *when* and *how often* it runs.
    async requestHandler(ctx) {
      const { request, page } = ctx;

      // Depth tracking is user-defined, but Crawlee persists userData
      // alongside requests for you.
      const depth = request.userData?.depth ?? 0;
      if (depth > opts.maxDepth) return;

      // Navigation timing is handled cooperatively:
      // Playwright does the waiting, Crawlee enforces the timeout.
      try {
        await page.waitForLoadState(opts.waitUntil, { timeout: opts.timeoutSecs * 1000 });
      } catch {}

      const url = request.loadedUrl || request.url;
      if (seen.has(url)) return;
      seen.add(url);

      // At this point, Crawlee has delivered a fully managed page.
      // We now hand off to *our* extraction logic.
      const extracted = await extractFromPage(page, opts);

      let html = null;
      if (opts.includeHtml) {
        html = await page.content();
      }

      // Result assembly is application-specific and outside Crawlee's scope.
      results.push({
        url,
        status: ctx.response?.status() ?? null,
        ...extracted,
        html
      });

      // Link scheduling is where Crawlee's queue matters most.
      // enqueueLinks handles normalization, deduplication, and scheduling.
      if (depth < opts.maxDepth && extracted.links.length) {
        await ctx.enqueueLinks({
          urls: extracted.links,
          transformRequestFunction: (req) => {
            const origin = toOrigin(req.url);
            if (allowedOrigin && origin !== allowedOrigin) return null;
            if (!req.url.startsWith("http")) return null;
            req.userData = { ...(req.userData || {}), depth: depth + 1 };
            return req;
          }
        });
      }
    }
  });

  // Logging configuration for Crawlee internals
  log.setLevel(log.LEVELS.INFO);

  // Crawl entrypoint. Crawlee now owns execution until completion.
  await crawler.run();

  // Final aggregation returned to the caller (e.g., LangGraph tool)
  return {
    startedAt: new Date().toISOString(),
    startUrls: opts.startUrls,
    maxPages: opts.maxPages,
    maxDepth: opts.maxDepth,
    sameOriginOnly: opts.sameOriginOnly,
    count: results.length,
    pages: results
  };
}
