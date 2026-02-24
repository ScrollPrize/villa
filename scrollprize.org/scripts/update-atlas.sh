#!/bin/bash

# Script to update the integrated atlas SPA from the vesuvius-atlas/browser/dist folder

set -e

# Configuration
ATLAS_DIST_DIR="/home/johannes/git/scrollprize/vesuvius-atlas/browser/dist"
TARGET_DIR="static/atlas"

echo "Updating atlas SPA integration..."

# Check if source directory exists
if [ ! -d "$ATLAS_DIST_DIR" ]; then
  echo "Error: Atlas dist directory not found at $ATLAS_DIST_DIR"
  echo "Please build the atlas first (cd to vesuvius-atlas/browser and run npm run build)"
  exit 1
fi

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Clean up old assets to avoid accumulation
echo "Cleaning old assets..."
if [ -d "$TARGET_DIR/assets" ]; then
  rm -rf "$TARGET_DIR/assets"
fi

# Copy all files
echo "Copying files from $ATLAS_DIST_DIR to $TARGET_DIR..."
cp -r "$ATLAS_DIST_DIR"/* "$TARGET_DIR/"

# Extract asset filenames from the built index.html
CSS_FILE=$(grep -oP 'href="/atlas/assets/\K[^"]+\.css' "$TARGET_DIR/index.html" || true)
JS_FILE=$(grep -oP 'src="/atlas/assets/\K[^"]+\.js' "$TARGET_DIR/index.html" || true)

if [ -n "$CSS_FILE" ] && [ -n "$JS_FILE" ]; then
  echo "Updating AtlasBrowser component with asset files: CSS=$CSS_FILE, JS=$JS_FILE"

  # Update the AtlasBrowser.js component with correct asset filenames
  COMPONENT_FILE="src/components/AtlasBrowser.js"
  if [ -f "$COMPONENT_FILE" ]; then
    # Update CSS filename
    sed -i "s|link.href = '/atlas/assets/.*\.css'|link.href = '/atlas/assets/$CSS_FILE'|g" "$COMPONENT_FILE"
    # Update JS filename
    sed -i "s|script.src = '/atlas/assets/.*\.js'|script.src = '/atlas/assets/$JS_FILE'|g" "$COMPONENT_FILE"
    echo "Component updated successfully"
  fi
fi

# Note: config.json now uses relative paths, no need to update

echo "Atlas integration updated successfully!"
echo "The atlas is now available at /atlas/"
