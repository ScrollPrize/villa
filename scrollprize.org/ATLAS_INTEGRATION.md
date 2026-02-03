# Atlas SPA Integration

The Vesuvius Atlas data browser is integrated into the Docusaurus site in two ways:

## Integration Points

### 1. Embedded in Docusaurus (`/data_browser`)
- Fully integrated with Docusaurus navigation and styling
- Located at: `docs/02_data_browser.mdx`
- Appears in the Data section of the sidebar
- Uses hash-based routing (e.g., `/data_browser#/samples/PHerc0123`)
- Inherits Docusaurus theme colors via CSS variables
- Shows only content (no header/footer) when embedded

### 2. Standalone SPA (`/atlas/`)
- Independent full-page experience
- Served from: `static/atlas/`
- Includes its own header and footer
- Can be used independently of the Docusaurus site

## How It Works

### Embedded Mode Detection
The atlas automatically detects when it's embedded in Docusaurus by checking if the `#atlas-root` container's parent is the document body:
- **Embedded**: Shows only content, uses Docusaurus layout
- **Standalone**: Shows full layout with header and footer

### Routing Strategy
- **Hash-based routing** (`HashRouter`) for maximum compatibility
- URLs like `/data_browser#/samples/PHerc0123` work with direct access
- No server-side routing configuration needed
- Deep linking works perfectly

### Styling Integration
The atlas inherits Docusaurus styles via CSS variables:
- Background: `--ifm-background-color`
- Text: `--ifm-font-color-base`
- Primary color: `--ifm-color-primary`
- Borders and grays: `--ifm-color-emphasis-*`

Tailwind config maps semantic colors to these variables, so the atlas automatically matches the site's dark theme.

## Files Location

- **Atlas source**: `/home/johannes/git/scrollprize/vesuvius-atlas/browser`
- **Built files**: `/home/johannes/git/scrollprize/vesuvius-atlas/browser/dist`
- **Static assets**: `static/atlas/`
- **Docusaurus component**: `src/components/AtlasBrowser.js`
- **Docusaurus page**: `docs/02_data_browser.mdx`

## Updating the Atlas

When you rebuild the atlas SPA, run the update script:

```bash
cd /home/johannes/git/scrollprize/vesuvius-atlas/browser
npm run build
cd /home/johannes/git/scrollprize/villa/scrollprize.org
./scripts/update-atlas.sh
```

The update script automatically:
1. Copies built files from `vesuvius-atlas/browser/dist` to `static/atlas/`
2. Extracts asset filenames from the build
3. Updates `AtlasBrowser.js` with correct JS/CSS references

## Configuration

### Atlas Configuration (`static/atlas/config.json`)
```json
{
  "metadataUrl": "metadata.json",
  "version": "0.1"
}
```

### Vite Configuration (`vesuvius-atlas/browser/vite.config.ts`)
```typescript
export default defineConfig({
  base: '/atlas/',
  plugins: [react()],
})
```

### Key Atlas Settings
- Uses `HashRouter` for client-side routing
- Loads config/metadata relative to `BASE_URL` (`/atlas/`)
- Detects embedded mode to conditionally render layout
- Inherits Docusaurus CSS variables for theming

## Development

### Local Development
```bash
npm start
```

Atlas available at:
- **Embedded**: `http://localhost:3000/data_browser`
- **Standalone**: `http://localhost:3000/atlas/`

### Production Deployment

No special server configuration needed! Hash-based routing means:
- `/data_browser` always loads the same page
- `/atlas/` serves static files normally
- Everything after `#` is handled client-side

## Technical Details

### Asset Loading
The `AtlasBrowser.js` component dynamically loads atlas assets:
```javascript
// Load CSS
const link = document.createElement('link');
link.href = '/atlas/assets/index-[hash].css';
document.head.appendChild(link);

// Load JS module
const script = document.createElement('script');
script.type = 'module';
script.src = '/atlas/assets/index-[hash].js';
document.body.appendChild(script);
```

### Container Setup
```javascript
<div id="atlas-root" className="dark">
  {/* Atlas mounts here */}
</div>
```

The `dark` class ensures dark mode is active, and the atlas CSS inherits Docusaurus variables.

## Maintenance

### Rebuilding After Atlas Changes
Any changes to the atlas source require rebuilding and updating:
```bash
cd /home/johannes/git/scrollprize/vesuvius-atlas/browser
npm run build
cd /home/johannes/git/scrollprize/villa/scrollprize.org
./scripts/update-atlas.sh
```

### Updating Styles
Atlas styling uses Tailwind with Docusaurus CSS variables. To modify:
1. Edit `vesuvius-atlas/browser/tailwind.config.js` for color mappings
2. Edit component classes to use semantic colors
3. Rebuild and update
