import React from "react";
import clsx from "clsx";
import { useActiveDocContext } from "@docusaurus/plugin-content-docs/client";
import DropdownNavbarItem from "@theme-original/NavbarItem/DropdownNavbarItem";

// Highlights a desktop navbar dropdown (e.g. "Tutorials & In-depth") on every
// page of the doc sidebar its items belong to, matching how `docSidebar`
// navbar items behave. Stock Docusaurus never marks a dropdown label active,
// so the section's archived pages would otherwise leave the navbar unlit.
export default function DropdownNavbarItemWrapper(props) {
  const { activeDoc, activeVersion } = useActiveDocContext(undefined);
  const sidebar = activeDoc?.sidebar;
  const active =
    !props.mobile &&
    !!sidebar &&
    (props.items ?? []).some((item) =>
      activeVersion?.docs.some((doc) => doc.sidebar === sidebar && doc.path === item.to),
    );
  return (
    <DropdownNavbarItem
      {...props}
      className={clsx(props.className, active && "navbar__link--active")}
    />
  );
}
