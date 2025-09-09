const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');

module.exports = function (context, options) {
  return {
    name: 'fetch-substack-posts',
    async loadContent() {
      return new Promise((resolve, reject) => {
        const scriptPath = path.join(__dirname, '../scripts/fetchLatestPosts.js');
        exec(`node ${scriptPath}`, (error, stdout, stderr) => {
          if (error) {
            console.error('Error fetching Substack posts:', error);
            console.error('stderr:', stderr);
            // Fail the build instead of continuing silently
            reject(new Error(`Failed to fetch Substack posts: ${error.message}`));
            return;
          }

          // Validate that the output file was created and is valid
          const outputPath = path.join(__dirname, '../static/data/latestPosts.json');

          try {
            if (!fs.existsSync(outputPath)) {
              throw new Error('Output file was not created');
            }

            const data = JSON.parse(fs.readFileSync(outputPath, 'utf8'));

            // Validate the structure
            if (!data.lastUpdated) {
              throw new Error('Missing lastUpdated field in output');
            }

            if (!Array.isArray(data.posts)) {
              throw new Error('Posts field is not an array');
            }

            // Check if there was an error in fetching
            if (data.error) {
              throw new Error(`RSS fetch failed: ${data.error}`);
            }

            // Validate that we have some posts (unless it's a legitimate empty feed)
            if (data.posts.length === 0) {
              console.warn('Warning: No posts were fetched from RSS feed');
            }

            // Validate post structure
            for (const post of data.posts) {
              if (!post.title || !post.link) {
                throw new Error('Invalid post structure: missing title or link');
              }
            }

            console.log(stdout.trim());
            console.log(`✓ Successfully validated ${data.posts.length} posts`);
            resolve(null);

          } catch (validationError) {
            console.error('Validation failed for Substack posts:', validationError.message);
            reject(new Error(`Substack posts validation failed: ${validationError.message}`));
          }
        });
      });
    },
  };
};
