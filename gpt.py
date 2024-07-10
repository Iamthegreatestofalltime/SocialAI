import jmespath
import asyncio
import json
from typing import Dict, List
from loguru import logger as log
from scrapfly import ScrapeConfig, ScrapflyClient, ScrapeApiResponse

SCRAPFLY = ScrapflyClient(key="scp-live-84ec83b142f94af8ba60ab0395960901")

js_scroll_function = """
function scrollToEnd(i) {
    // check if already at the bottom and stop if there aren't more scrolls
    if (window.innerHeight + window.scrollY >= document.body.scrollHeight) {
        console.log("Reached the bottom.");
        return;
    }

    // scroll down
    window.scrollTo(0, document.body.scrollHeight);

    // set a maximum of 15 iterations
    if (i < 15) {
        setTimeout(() => scrollToEnd(i + 1), 3000);
    } else {
        console.log("Reached the end of iterations.");
    }
}

scrollToEnd(0);
"""

def parse_channel(response: ScrapeApiResponse):
    """parse channel video data from XHR calls"""
    # extract the xhr calls and extract the ones for videos
    xhr_calls = response.scrape_result["browser_data"]["xhr_call"]
    post_calls = [c for c in xhr_calls if "/api/post/item_list/" in c["url"]]
    post_data = []
    for post_call in post_calls:
        try:
            data = json.loads(post_call["response"]["body"])["itemList"]
        except Exception as e:
            log.error(f"Post data couldn't load: {e}")
            continue
        post_data.extend(data)
    # parse all the data using jmespath
    parsed_data = []
    for post in post_data:
        result = jmespath.search(
            """{
            createTime: createTime,
            desc: desc,
            id: id,
            stats: stats,
            contents: contents[].{desc: desc, textExtra: textExtra[].{hashtagName: hashtagName}},
            video: video
            }""",
            post
        )
        parsed_data.append(result)    
    return parsed_data


async def scrape_channel(url: str) -> List[Dict]:
    """scrape video data from a channel (profile with videos)"""
    log.info(f"scraping channel page with the URL {url} for post data")
    response = await SCRAPFLY.async_scrape(ScrapeConfig(
        url, asp=True, country="GB", render_js=True, rendering_wait=2000, js=js_scroll_function
    ))
    data = parse_channel(response)
    log.success(f"scraped {len(data)} posts data")
    return data

async def run():
    channel_data = await scrape_channel(
        url="https://www.tiktok.com/@essentialdetailingcarmel"
    )
    # save the result to a JSON file
    with open("channel_data.json", "w", encoding="utf-8") as file:
        json.dump(channel_data, file, indent=2, ensure_ascii=False)
    
    # Print the first few video IDs
    for i, post in enumerate(channel_data[:5]):  # Print first 5 video IDs
        print(f"Video {i+1} ID: {post['id']}")

if __name__ == "__main__":
    asyncio.run(run())