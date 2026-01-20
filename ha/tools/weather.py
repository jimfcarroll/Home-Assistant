def get_weather(location: str) -> str:
    """
    Look up the weather

    Guidance:
    - Format your query starting with the intent first, then the context.
    - You will get a list of URLs back.
    - Use the URLs to read web pages with `read_web_page`.
    """
    print(f"LLM passed:\n{location}")
    return (
        "Tomorrow will be partly cloudy. "
        "High temperature will be seventy two degrees Fahrenheit. "
        "Low temperature will be fifty five degrees Fahrenheit."
    )
