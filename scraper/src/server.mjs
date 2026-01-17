import express from "express";
import { z } from "zod";
import { runCrawl } from "./crawl.mjs";

const app = express();
app.use(express.json({ limit: "2mb" }));

app.get("/health", (_req, res) => res.json({ ok: true }));

const CrawlRequest = z.object({
  startUrls: z.array(z.string().url()).min(1),
  maxPages: z.number().int().min(1).max(500).default(25),
  maxDepth: z.number().int().min(0).max(10).default(2),
  sameOriginOnly: z.boolean().default(true),
  includeHtml: z.boolean().default(false),
  includeLinks: z.boolean().default(true),
  userAgent: z.string().min(1).optional(),
  timeoutSecs: z.number().int().min(5).max(120).default(30),
  waitUntil: z.enum(["load", "domcontentloaded", "networkidle"]).default("domcontentloaded"),
  concurrency: z.number().int().min(1).max(10).default(3)
});

app.post("/crawl", async (req, res) => {
  const parsed = CrawlRequest.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ error: "Invalid request", details: parsed.error.flatten() });
  }

  try {
    const result = await runCrawl(parsed.data);
    res.json(result);
  } catch (err) {
    res.status(500).json({
      error: "Crawl failed",
      message: err?.message ?? String(err)
    });
  }
});

const port = Number(process.env.PORT || 3000);
app.listen(port, () => {
  console.log(`scraper listening on :${port}`);
});
