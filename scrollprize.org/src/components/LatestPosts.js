import React, { useState, useEffect } from "react";

// Dense "Updates" strip: 1 permanent "Get Started" entry + 3 latest Substack
// posts, one compact row per entry (title 15px/600 + date 12px faint).
// Layout/skin lives in chrome.css (.vc-updates): 4-up >=1280px, 2-up >=480px,
// single column below 480px, hairline separators, no shadows.
// Fetches posts from /data/latestPosts.json (generated at build time by fetchLatestPosts.js)
// RSS feed is fetched from https://scrollprize.substack.com/feed during build

// Fallback posts in case RSS fetch fails
const FALLBACK_POSTS = [
  {
    title: "$60,000 First Title Prize Awarded",
    href: "https://scrollprize.substack.com/p/60000-first-title-prize-awarded",
    subtext: "May 5"
  },
  {
    title: "February Prizes and Updates",
    href: "https://scrollprize.substack.com/p/february-progress-prizes-and-updates",
    subtext: "March 12"
  },
  {
    title: "New Prizes and Progress Update",
    href: "https://scrollprize.substack.com/p/new-prizes-and-an-update-on-progress",
    subtext: "February 27"
  }
];

const formatDate = (dateString) => {
  const date = new Date(dateString);
  const months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  return `${months[date.getMonth()]} ${date.getDate()}`;
};

const LatestPosts = () => {
  const [posts, setPosts] = useState(FALLBACK_POSTS);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Try to load the generated posts data
    const loadPosts = async () => {
      try {
        const response = await fetch('/data/latestPosts.json');
        if (response.ok) {
          const data = await response.json();
          if (data.posts && data.posts.length > 0) {
            // Format the posts data
            const formattedPosts = data.posts.slice(0, 3).map(post => ({
              title: post.title,
              href: post.link,
              subtext: formatDate(post.pubDate)
            }));
            setPosts(formattedPosts);
          }
        }
      } catch (error) {
        console.error('Failed to load latest posts:', error);
        // Keep using fallback posts
      } finally {
        setLoading(false);
      }
    };

    loadPosts();
  }, []);

  return (
    <div className="vc-updates mb-3" role="list">
      <a className="vc-updates__item" href="/get_started" role="listitem">
        <span className="vc-updates__title">Get Started</span>
        <span className="vc-updates__meta">$1.8M+ already awarded</span>
      </a>
      {posts.map((post, index) => (
        <a
          key={index}
          className="vc-updates__item"
          href={post.href}
          role="listitem"
        >
          <span className="vc-updates__title">{post.title}</span>
          <span className="vc-updates__meta">{post.subtext}</span>
        </a>
      ))}
    </div>
  );
};

export default LatestPosts;
