const Parser = require('rss-parser');
const fs = require('fs');
const path = require('path');

const RSS_URL = 'https://scrollprize.substack.com/feed';
const OUTPUT_PATH = path.join(__dirname, '../static/data/latestPosts.json');

async function fetchLatestPosts() {
  const parser = new Parser();
  
  try {
    console.log('Fetching latest posts from Substack RSS feed...');
    const feed = await parser.parseURL(RSS_URL);
    
    // Extract the latest posts (excluding any that might be pinned or special)
    const posts = feed.items.slice(0, 10).map(item => ({
      title: item.title,
      link: item.link,
      pubDate: item.pubDate,
      description: item.contentSnippet || item.content || ''
    }));
    
    // Create directory if it doesn't exist
    const outputDir = path.dirname(OUTPUT_PATH);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    // Write the data to a JSON file
    const data = {
      lastUpdated: new Date().toISOString(),
      posts: posts
    };
    
    fs.writeFileSync(OUTPUT_PATH, JSON.stringify(data, null, 2));
    console.log(`Successfully fetched ${posts.length} posts and saved to ${OUTPUT_PATH}`);
    
  } catch (error) {
    console.error('Error fetching RSS feed:', error);
    // Create a fallback file with empty posts array
    const fallbackData = {
      lastUpdated: new Date().toISOString(),
      posts: [],
      error: error.message
    };
    
    // Ensure directory exists
    const outputDir = path.dirname(OUTPUT_PATH);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    fs.writeFileSync(OUTPUT_PATH, JSON.stringify(fallbackData, null, 2));
  }
}

// Run the script
fetchLatestPosts();
