import React, { useState, useEffect } from "react";
import TopCard from "./TopCard";

// Displays 4 TopCards: 1 permanent "Get Started" + 3 latest Substack posts
// Fetches posts from /data/latestPosts.json (generated at build time by fetchLatestPosts.js)
// RSS feed is fetched from https://scrollprize.substack.com/feed during build

// Fallback posts in case RSS fetch fails
const FALLBACK_POSTS = [
  {
    title: "We are cooking",
    href: "https://scrollprize.substack.com/p/we-are-cooking",
    subtext: "March 19"
  },
  {
    title: "~70% of PHerc. 172 is now digitally unwrapped",
    href: "https://scrollprize.substack.com/p/70-of-pherc-172-is-now-digitally",
    subtext: "January 13"
  },
  {
    title: "Finally—letters in Scroll 4!",
    href: "https://scrollprize.substack.com/p/finallyletters-in-scroll-4",
    subtext: "December 21"
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
    <div className="grid grid-cols-2 xl:grid-cols-4 gap-4 max-w-9xl pb-3">
      <TopCard
        title="Get Started"
        href="/get_started"
        subtext="$1.78M+ already awarded"
        useArrow={true}
      />
      {posts.map((post, index) => (
        <TopCard
          key={index}
          title={post.title}
          href={post.href}
          subtext={post.subtext}
        />
      ))}
    </div>
  );
};

export default LatestPosts;
