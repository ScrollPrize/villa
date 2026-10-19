# VC3D themes

VC3D's look comes from a theme. Three are built in:

| Id | Name | What it is |
|---|---|---|
| `classic` | Classic | VC3D's original look and the default: the native style, a dark palette when the system prefers dark, and the gradient menu bar. |
| `modern-dark` | Modern Dark | Qt's Fusion style with dark panels and one accent color. |
| `modern-light` | Modern Light | Fusion with light panels and one accent color. |

Anyone can write a theme as a small JSON file, share it, and import it. A
theme restyles the application chrome (menus, panels, buttons, inputs).
Volume views, overlays and widgets that set their own colors keep them.

## Picking a theme

- **View > Theme** lists the built-in themes, then your own sorted by name.
  Picking one applies it at once, without a restart, and VC3D starts with it
  next time. Hover a theme to see its author, description and file.
- **`--theme <id or path>`** on the command line uses a theme for that run
  only, without saving it. It takes a theme id or the path of a theme file or
  folder, which is loaded as is without being installed. Handy for
  screenshots and scripts:

  ```sh
  VC3D --theme modern-dark
  VC3D --theme ~/Downloads/ocean-night.vc3dtheme
  ```

- **`VC3D_THEME`** does the same from the environment.

The choice is saved as `appearance/theme` in `~/.VC3D/VC3D.ini` (or
`$VC3D_CONFIG_DIR/VC3D.ini`), as a theme id or, for a theme picked by path,
as that path. At startup VC3D tries `--theme`, then `VC3D_THEME`, then the
saved theme, then Classic, and uses the first that works. Each one that fails
is reported on the console and in the status bar.

The Theme submenu also has:

- **Import Theme File...** validates a `.vc3dtheme` file, copies it into your
  themes folder and switches to it. The message names the installed copy;
  edit that copy to change the theme.
- **Import Theme Folder...** does the same for a folder theme: pick the folder
  that holds `theme.vc3dtheme`, and the whole folder is copied.
- **Export Current Theme as Template...** writes the current theme as a
  complete theme file to edit (see [Making a theme](#making-a-theme)).
- **Open Themes Folder** opens your themes folder, creating it if needed.
- **Reload Themes** rescans the folders and lists any theme that has problems.

VC3D asks before an import replaces a theme you already have with the same
id, and keeps the installed one if the new one cannot be installed.

## Where themes live

VC3D looks in these places, in this order, and uses the first theme it finds
for each id:

1. The built-in themes.
2. Your themes folder:

   | System | Folder |
   |---|---|
   | Linux | `~/.local/share/Vesuvius Challenge/VC3D/themes` |
   | macOS | `~/Library/Application Support/Vesuvius Challenge/VC3D/themes` |
   | Windows | `%APPDATA%\Vesuvius Challenge\VC3D\themes` |

3. Each folder in `VC3D_THEME_PATH`, separated by `:` (`;` on Windows).

A theme in a folder is either a single `<id>.vc3dtheme` file or a folder
`<id>/` holding `theme.vc3dtheme` plus its images and style sheets. The id is
the file or folder name in lowercase, with every run of other characters
turned into one dash: `Ocean Night.vc3dtheme` becomes `ocean-night`. A user
theme cannot use a built-in id; such files are skipped with a warning. When
two folders hold the same id, the first one in the order above wins and the
other is skipped with a warning.

Themes draw their check marks and arrows into
`~/.cache/Vesuvius Challenge/VC3D/themes` (Linux) or
`~/Library/Caches/Vesuvius Challenge/VC3D/themes` (macOS), one small folder
per theme and set of colors. Folders for a theme's earlier colors are removed
after a week. The folder is safe to delete.

### Sharing themes in a lab

Put the theme files in a shared folder and point everyone's VC3D at it:

```sh
export VC3D_THEME_PATH=/lab/shared/vc3d-themes
```

Themes from `VC3D_THEME_PATH` appear in View > Theme without being copied.
To share a single theme, send the file (or the zipped folder) and import it.

## Themes from other people

A theme can restyle, hide or shrink anything VC3D draws with Qt style sheets,
and show images from its own folder. It cannot run code or change what
controls do: VC3D refuses a theme whose style sheet sets widget properties
(`qproperty-*`), loads images from outside the theme's folder, links or
special files, does not parse, or is larger than the limits (64 KiB per theme
file or style sheet, 4 MiB per image). Anyone who can write to a
`VC3D_THEME_PATH` folder decides the look for everyone who uses it.

If a theme makes VC3D hard to use, start it with Classic for one run and then
pick another theme or remove the bad one from your themes folder:

```sh
VC3D --theme classic
```

`VC3D_THEME=classic` works the same, and deleting the `theme=` line under
`[appearance]` in `VC3D.ini` makes Classic the saved choice again.

## Making a theme

1. Pick the built-in theme closest to what you want: Modern Dark for a dark
   theme, Modern Light for a light one.
2. Choose **View > Theme > Export Current Theme as Template...**. The dialog
   opens in your themes folder; name the file, for example
   `ocean-night.vc3dtheme`. The file name becomes the theme's id.
3. The new theme is listed in View > Theme right away. Pick it.
4. Edit the file and save. VC3D reapplies the theme each time (see
   [Editing a theme live](#editing-a-theme-live)).
5. Share the file, or the theme's folder if it has images.

The exported file lists every token and the palette roles no token covers,
so you only change values. Exporting Classic gives a Modern Dark template,
since Classic has no tokens. To edit a file outside your themes folder, run
`VC3D --theme /path/to/ocean-night.vc3dtheme`; it is watched the same way.

### Change the accent

The accent appears in seventeen tokens. Set all of them, or the old accent
shows through in focus outlines and selections:

```json
{
  "vc3dTheme": 1,
  "name": "Ocean Night",
  "author": "Jane Doe",
  "description": "Modern Dark with an orange accent.",
  "base": "modern-dark",
  "tokens": {
    "accent": "#ff8a3d",
    "accent_fill": "#b45309",
    "accent_hover": "#a64b08",
    "accent_pressed": "#92400e",
    "accent_disabled": "#4a3322",
    "on_accent": "#ffffff",
    "on_accent_disabled": "#d6c7bb",
    "accent_soft": "#4a2a14",
    "accent_soft_hover": "#5c3519",
    "accent_soft_border": "#9a5427",
    "accent_soft_text": "#fff1e6",
    "item_selected": "#5e3417",
    "focus": "#ff8a3d",
    "focus_soft": "#ff8a3d",
    "focus_on_accent": "#fff1e6",
    "selection": "#b45309",
    "selection_text": "#ffffff"
  }
}
```

That file works as is: save it as `ocean-night.vc3dtheme` in your themes
folder and choose Reload Themes.

When you pick your own colors, keep text on `accent_soft` and
`item_selected` above 4.5:1, `on_accent` on `accent_fill` above 4.5:1, and
`accent` at least 3:1 against `border` so checked toggles stand out, or make
the change visible another way, for example by switching from gray to a
bright accent.

### The format

A `.vc3dtheme` file is JSON. Every key:

```json
{
  "vc3dTheme": 1,
  "$comment": "Notes for people who edit the file; VC3D ignores this key.",
  "name": "Ocean Night",
  "author": "Jane Doe",
  "description": "Navy panels with an orange accent.",
  "base": "modern-dark",
  "colorScheme": "dark",
  "tokens": { "window": "#0f1b2d", "accent": "#ff8a3d" },
  "palette": { "LinkVisited": "#c084fc", "disabled": { "Text": "#6b7280" } },
  "font": { "family": "Inter", "pointSize": 10 },
  "stylesheet": "extra.qss",
  "stylesheetMode": "append"
}
```

| Key | Required | Meaning |
|---|---|---|
| `vc3dTheme` | yes | Format version, the integer `1`. A file with a higher version is refused with a message to update VC3D. |
| `$comment` | no | Free text for people editing the file. |
| `name` | yes | Name shown in the Theme menu: one line, at most 80 characters. |
| `author`, `description` | no | Shown in the menu's tool tip: plain text, at most 80 and 500 characters. |
| `base` | no | Theme to start from: a built-in id or another user theme's id. Default `modern-dark`. Bases can chain up to 8 deep; a cycle is an error. |
| `colorScheme` | no | `dark` or `light`. Default: the base's. On Qt 6.8 and newer it also sets the application color scheme. For a Classic-based theme it picks Classic's dark or light variant instead of following the system. |
| `tokens` | no | Token values by name: colors, lengths and the `mono` font list; see [Tokens](#tokens). Tokens you leave out come from the base. |
| `palette` | no | Explicit Qt palette roles; see [Palette](#palette). |
| `font` | no | Application font: `family` and/or `pointSize` (above 0, at most 96). A family that is not installed is ignored and the current font kept. |
| `stylesheet` | no | Extra Qt style sheet file, relative to the theme file and inside the theme's folder. |
| `qss` | no | Extra style sheet as inline text. Use `stylesheet` or `qss`, not both. |
| `stylesheetMode` | no | `append` (default) or `replace`; see [Extra style sheets](#extra-style-sheets). |

Unknown keys and unknown token names are warnings, not errors, so a theme
written for a newer VC3D still loads; a likely misspelling comes with a
suggestion, such as `unknown token "acent" (did you mean "accent"?)`. A token
the theme's own style sheet uses is not reported. JSON allows no comments,
trailing commas or single quotes.

Colors are `#rgb`, `#rrggbb`, `#aarrggbb` or an SVG color name such as
`black`. Qt puts the alpha first: `#80ff8a3d` is a half transparent orange,
while `#ff8a3d80` is an opaque purple, so VC3D warns about eight digit
colors. `rgb()` is not accepted. Lengths are a number with `px`, `pt`, `em`
or `ex`, or a bare number.

With `"base": "classic"` the theme keeps Classic's native style and gradient
menu bar and the built-in template is not used; such a theme can only add
`palette`, `font` and its own style sheet.

### Tokens

The Modern themes paint the application from one style sheet template whose
`@token@` placeholders are filled from these tokens. Most are colors; the
Shape and Type tokens are lengths in pixels (`6px`) and `mono` is a font list.
A theme may override any of them. The table gives the values of both
built-in themes; the files are
[`builtin/modern-dark.vc3dtheme`](../apps/VC3D/themes/builtin/modern-dark.vc3dtheme)
and
[`builtin/modern-light.vc3dtheme`](../apps/VC3D/themes/builtin/modern-light.vc3dtheme).

| Token | Modern Dark | Modern Light | What it colors or sizes |
|---|---|---|---|
| **Surfaces** | | | |
| `chrome` | `#11151b` | `#e9edf2` | Menu bar, the workspace tab row and the gaps between docks: the frame around the work area. |
| `window` | `#161b22` | `#f4f6f9` | Main surface: dock panels, dialogs, status bar (QPalette Window). |
| `section` | `#1c232c` | `#eceff4` | Header of a collapsible settings section in a dock panel. |
| `section_body` | `#191f27` | `#fafbfc` | Body of an expanded collapsible section and of a group box: a card one step above the panel. |
| `raised` | `#1f2631` | `#ffffff` | Floating surfaces: menus, combo box lists, tool tips (QPalette ToolTipBase). |
| `header` | `#1a2029` | `#f0f3f6` | Column and row headers of tables and trees. |
| `input` | `#0e1217` | `#ffffff` | Fields and lists: line edits, spin boxes, combo boxes, item views (QPalette Base). |
| `input_disabled` | `#141920` | `#eff1f4` | Disabled or read-only fields. |
| `alt_row` | `#12171d` | `#f7f9fb` | Alternate rows in lists and tables (QPalette AlternateBase). |
| **Controls** | | | |
| `button` | `#212935` | `#ffffff` | Push button and tool button at rest (QPalette Button). |
| `button_hover` | `#29323f` | `#f0f3f7` | Push button under the pointer. |
| `button_pressed` | `#313b4a` | `#e3e8ef` | Push button while pressed; tool button while pressed. |
| `button_disabled` | `#1a1f27` | `#f1f3f6` | Disabled push button. |
| `hover` | `#232b36` | `#e5eaf0` | Hovered rows, menu bar items, tabs, flat buttons, dock bar buttons and spin box steps. |
| `scroll_handle` | `#5a687c` | `#8290a1` | Scroll bar handle (at least 3:1 against the window). |
| `scroll_handle_hover` | `#6e7d93` | `#6b7685` | Scroll bar handle under the pointer. |
| `slider_track` | `#2a3341` | `#d5dce5` | Slider groove. |
| `slider_handle` | `#e4e9f0` | `#ffffff` | Slider handle fill (ringed with the accent). |
| `progress_track` | `#0e1217` | `#ffffff` | Empty part of a progress bar. |
| `progress_chunk` | `#0e7490` | `#2b9bb3` | Filled part of a progress bar (at least 3:1 against progress_track). |
| `progress_text` | `#ffffff` | `#17202b` | Percentage text of a progress bar; drawn over both parts, so it needs 4.5:1 on each. |
| **Lines** | | | |
| `border_subtle` | `#232b36` | `#e0e5eb` | Hairlines: separators, group box and section outlines, splitters, disabled outlines. |
| `border` | `#313b49` | `#ccd4de` | Outlines of buttons and menus; disabled fields and check boxes. |
| `field_border` | `#5b6a7f` | `#7a8899` | Outlines of fields, lists, check boxes and tool tips (at least 3:1 against the window, so fields can be found). |
| `border_strong` | `#718099` | `#5f6b7c` | Outlines under the pointer (buttons, fields, check boxes). |
| **Text** | | | |
| `text` | `#e4e9f0` | `#17202b` | Primary text (QPalette Text, WindowText, ButtonText); group box titles. |
| `text_secondary` | `#9ba7b6` | `#556173` | Labels of lesser weight: menu bar, tabs, status bar, dock bar, the project name. |
| `text_muted` | `#7d8a9c` | `#6a7484` | Disabled text and placeholders (QPalette PlaceholderText). |
| **Accent** | | | |
| `accent` | `#22d3ee` | `#0e7490` | The line accent: the selected tab underline, slider fill and handle ring, hovered check box outline, checked toggle outline and text, the selected row bar, menu check marks. |
| `accent_fill` | `#0e7490` | `#0e7490` | The fill accent, calmer than accent: checked check boxes and radio buttons, the default button (OK). |
| `accent_hover` | `#0f7d9b` | `#155e75` | The default button under the pointer (a step from accent_fill). |
| `accent_pressed` | `#0b6178` | `#164e63` | The default button while pressed. |
| `accent_disabled` | `#2a4e57` | `#a9d6e2` | Fill of a disabled checked check box or radio button. |
| `on_accent` | `#ffffff` | `#ffffff` | Text and check marks drawn on accent_fill. |
| `on_accent_disabled` | `#b8c2ce` | `#3f4a58` | Check mark, dash and radio dot of a disabled checked control, on accent_disabled. |
| `accent_soft` | `#133e4a` | `#d4f1f8` | Soft tint for the checked state: menu items, checked buttons, the open dock panel's button. |
| `accent_soft_hover` | `#18505e` | `#c2eaf4` | Checked button or highlighted menu item under the pointer. |
| `accent_soft_border` | `#1f6b7c` | `#7cc9db` | Outline of the open dock panel's button; pressed scroll handle; splitter and dock gap hover. |
| `accent_soft_text` | `#e8fbff` | `#06323c` | Text on accent_soft and on item_selected. |
| `item_selected` | `#1b5c6e` | `#a8dcec` | Selected rows in lists, trees, tables and combo box lists. |
| `focus` | `#22d3ee` | `#0e7490` | Keyboard focus: outline of the focused field, check box, tab, list item. |
| `focus_soft` | `#22d3ee` | `#0e7490` | Outline of the focused push button and tool button. |
| `focus_on_accent` | `#e4e9f0` | `#17202b` | Focus outline on accent_fill (the focused default button, a focused checked box). |
| `selection` | `#0e7490` | `#0e7490` | Selected text in fields (QPalette Highlight). |
| `selection_text` | `#ffffff` | `#ffffff` | Selected text color (QPalette HighlightedText). |
| **Status** | | | |
| `warning_text` | `#664d03` | `#664d03` | Text and button text on VC3D's yellow warning banner (the banner keeps its own background). |
| `warning_border` | `#9a7a12` | `#9a7a12` | Outline of the button on the warning banner. |
| `danger` | `#f87171` | `#b91c1c` | Error text: labels whose dynamic property vc3dStatus is "danger" (at least 4.5:1 on window, section_body and input). |
| `warning` | `#fbbf24` | `#b45309` | Warning text: labels whose vc3dStatus is "warning". |
| `success` | `#4ade80` | `#15803d` | Success text: labels whose vc3dStatus is "success". |
| **Shape** | | | |
| `radius_small` | `4px` | `4px` | Corners of check boxes, list and menu items, scroll handles, progress bars. |
| `radius` | `6px` | `6px` | Corners of buttons, fields, sections, group boxes, tabs, dock bar buttons. |
| `radius_large` | `8px` | `8px` | Corners of tab panes. |
| `radius_popup` | `0px` | `0px` | Corners of menus, tool tips and combo box lists. These are separate windows that Qt fills edge to edge, so anything above 0px shows square corners unless VC3D makes pop-ups translucent. |
| **Type** | | | |
| `font_size_small` | `12px` | `12px` | Text size of the bottom dock bar buttons. |
| `mono` | see below | see below | Font stack for code and logs in plain text views (JSON profiles, the console, fiber trace details). |

`mono` is `"JetBrains Mono", "SF Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace`.

VC3D also fills in these placeholders; themes cannot define them:

| Placeholder | Value |
|---|---|
| `@name@` | The folder of this theme's drawn icons, used as `url(vc3dtheme:@name@/check.png)`. |
| `@scheme@` | `dark` or `light`. |
| `@themedir@` | The theme's own folder, for images it ships: `url(@themedir@/grain.png)`. Quote the URL if the path may contain spaces: `url("@themedir@/grain.png")`. |

The template draws these icons in the theme's colors, each at 1x and 2x:
`check`, `check-disabled`, `dash`, `dash-disabled`, `menu-check`,
`chevron-down`, `chevron-down-disabled`, `chevron-down-hover`,
`chevron-right`, `chevron-right-selected`, `spin-up`, `spin-down`,
`spin-up-disabled`, `spin-down-disabled`, `spin-up-hover`, `spin-down-hover`
(`.png`).

### Palette

Besides the style sheet, Qt widgets that paint themselves read the
application palette. A Modern-based theme gets its palette from its tokens:

| Role | From | Role | From |
|---|---|---|---|
| Window | `window` | Button | `button` |
| WindowText | `text` | ButtonText | `text` |
| Base | `input` | Highlight | `selection` |
| AlternateBase | `alt_row` | HighlightedText | `selection_text` |
| ToolTipBase | `raised` | Link | `accent` |
| ToolTipText | `text` | Mid | `border` |
| PlaceholderText | `text_muted` | Light, Midlight | `border`, lighter |
| Text | `text` | Dark, Shadow | `window`, darker |

In the disabled group, Text, WindowText and ButtonText come from
`text_muted`, Base from `input_disabled` and Button from `button_disabled`.
On Qt 6.6 and newer, Accent comes from `accent`.

`palette` sets roles explicitly, by Qt's role name (case does not matter):
Window, WindowText, Base, AlternateBase, ToolTipBase, ToolTipText,
PlaceholderText, Text, Button, ButtonText, BrightText, Light, Midlight, Mid,
Dark, Shadow, Highlight, HighlightedText, Link, LinkVisited (and Accent on
Qt 6.6+). A role applies to every color group; the `disabled` object
overrides the disabled group. The built-in Modern themes list only the roles
no token covers (BrightText, Light, Midlight, Mid, Dark, Shadow, LinkVisited
and a few disabled roles). A theme that changes a token still recolors the
roles that come from it, even when its base set them explicitly: changing
`window` changes Window.

### Extra style sheets

A theme can add its own Qt style sheet, from a file (`stylesheet`) or inline
(`qss`). It gets the same `@token@` substitution as the template.

- `"stylesheetMode": "append"` (the default) adds it after the template's
  output, so its rules win where they overlap.
- `"stylesheetMode": "replace"` drops the template: the theme supplies the
  entire style sheet. VC3D still uses Fusion, the palette, the font and the
  drawn icons, but nothing else. For power users.

Extra style sheets of base themes are kept, in order, so a theme based on
yours looks like yours plus its own changes. Qt ignores a whole style sheet
that does not parse, so VC3D checks that braces, comments and strings are
closed and names the line of the first problem. An unknown placeholder is an
error; text inside comments is not checked. A style sheet may only style
widgets: `qproperty-*` is refused, and `url()` may only name the drawn icons
(`vc3dtheme:`), Qt resources (`:/`) or regular files inside the theme's
folder, written as `url(@themedir@/name.png)`. A style sheet that uses
`@themedir@` images needs a folder theme, so the images travel with it;
importing a single file that uses them is refused.

### Editing a theme live

While your theme is the active one, VC3D watches its file and its style sheet
file. Save, and VC3D reapplies the theme about 200 ms later; editors that save
by replacing the file are handled, and other files changing in the same
folder are ignored. If an edit is invalid, VC3D keeps the last good version,
says why in the status bar, and prints the details on the console once.
Warnings, such as a misspelled token, also show in the status bar.

### Limits

- A theme that sets a font makes the application font explicit: after
  switching back to a theme without one, VC3D uses its startup font but no
  longer follows system font changes until it restarts.
- A few widgets pick their colors when they are created, such as the ring of
  the range slider handles, and keep VC3D's startup colors until it restarts.
- A style sheet given with Qt's `-stylesheet` option applies under Classic
  and Classic-based themes; the Modern themes replace it.

## For developers

- Code: `apps/VC3D/themes/ThemeEngine.cpp` (parsing, validation, base
  chains, template rendering, style sheet checks, icons, import and export),
  `ThemeManager.cpp` (applying themes to the running application, discovery,
  persistence and live reload) and `ClassicTheme.cpp` (the classic look).
- The built-in Modern themes are `themes/builtin/modern-dark.vc3dtheme` and
  `themes/builtin/modern-light.vc3dtheme`; both render
  `themes/builtin/theme.qss.in`. They are compiled in through `themes.qrc` as
  `:/themes/...`. The token names in `builtin/modern-dark.vc3dtheme` define
  which tokens are known. The icons the template refers to are drawn from
  the `kIconRecipes` table in `ThemeEngine.cpp`.
- Tests: `apps/VC3D/test/test_theme_engine.cpp` (QtTest, offscreen) checks
  that every template placeholder and icon is known and that every built-in
  token is listed in this page's token table.
