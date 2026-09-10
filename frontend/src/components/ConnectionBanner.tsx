import { useEffect, useState } from "react";

/** Backend erişilemezse üstte uyarı şeridi gösterir. */
export default function ConnectionBanner() {
  const [down, setDown] = useState(false);

  useEffect(() => {
    let active = true;
    const check = async () => {
      try {
        const r = await fetch("/api/health", { cache: "no-store" });
        if (active) setDown(!r.ok);
      } catch {
        if (active) setDown(true);
      }
    };
    check();
    const id = setInterval(check, 10000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  if (!down) return null;
  return (
    <div className="bg-amber-500 px-4 py-2 text-center text-sm font-medium text-amber-950">
      ⚠ Yapay zekâ servisine (backend :8000) ulaşılamıyor. Sunucunun çalıştığından
      emin olun — <code className="rounded bg-amber-600/30 px-1">./run_app.sh</code>
    </div>
  );
}
