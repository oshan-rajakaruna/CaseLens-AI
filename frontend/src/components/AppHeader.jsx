const navigationItems = [
  { id: "home", label: "Home", href: "/" },
  { id: "dashboard", label: "Dashboard", href: "/?page=dashboard" },
  { id: "cases", label: "Cases", href: "/?page=cases" },
  { id: "reports", label: "Reports", href: "/?page=reports" },
];

export default function AppHeader({ activePage = "home" }) {
  return (
    <header className="app-header">
      <a className="brand" href="/" aria-label="CaseLens home">
        CaseLens
      </a>
      <nav aria-label="Primary navigation">
        <ul className="workspace-navigation">
          {navigationItems.map((item) => (
            <li key={item.id}>
              <a
                aria-current={activePage === item.id ? "page" : undefined}
                href={item.href}
              >
                {item.label}
              </a>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  );
}
