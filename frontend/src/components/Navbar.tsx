import { NavLink } from "react-router-dom";
import ThemeToggle from "./ThemeToggle";

const links = [
  { to: "/", label: "Genel Bakış", end: true },
  { to: "/predictor", label: "Tahmin" },
  { to: "/critical-pga", label: "Kritik PGA" },
  { to: "/map", label: "Harita" },
  { to: "/methodology", label: "Yöntem" },
  { to: "/validation", label: "Doğrulama" },
  { to: "/results", label: "Sonuçlar" },
];

export default function Navbar() {
  return (
    <header className="sticky top-0 z-[1000] border-b border-slate-200 bg-white/80 backdrop-blur dark:border-slate-800 dark:bg-slate-950/80">
      <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3">
        <div className="flex items-center gap-2.5">
          <div className="grid h-9 w-9 place-items-center rounded-xl bg-brand-600 text-white shadow-sm">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 20h18L14 6l-4 7-3-4-4 11z" />
            </svg>
          </div>
          <div className="leading-tight">
            <div className="text-sm font-bold tracking-tight text-slate-900 dark:text-white">
              Heyelan Dijital Twin
            </div>
            <div className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
              Kahramanmaraş · 6 Şubat 2023
            </div>
          </div>
        </div>

        <nav className="ml-auto hidden items-center gap-1 md:flex">
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.end}
              className={({ isActive }) =>
                `nav-link ${isActive ? "nav-link-active" : ""}`
              }
            >
              {l.label}
            </NavLink>
          ))}
        </nav>

        <div className="ml-auto md:ml-2">
          <ThemeToggle />
        </div>
      </div>

      {/* mobil menü */}
      <nav className="flex items-center gap-1 overflow-x-auto border-t border-slate-100 px-4 py-2 md:hidden dark:border-slate-800">
        {links.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            end={l.end}
            className={({ isActive }) =>
              `nav-link whitespace-nowrap ${isActive ? "nav-link-active" : ""}`
            }
          >
            {l.label}
          </NavLink>
        ))}
      </nav>
    </header>
  );
}
