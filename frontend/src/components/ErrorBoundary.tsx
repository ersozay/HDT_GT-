import { Component, ErrorInfo, ReactNode } from "react";

export default class ErrorBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean; msg?: string }
> {
  state = { hasError: false, msg: undefined as string | undefined };

  static getDerivedStateFromError(err: Error) {
    return { hasError: true, msg: err.message };
  }

  componentDidCatch(err: Error, info: ErrorInfo) {
    console.error("UI hata sınırı:", err, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="mx-auto max-w-lg py-20 text-center">
          <div className="mx-auto mb-4 grid h-14 w-14 place-items-center rounded-2xl bg-rose-100 text-rose-600 dark:bg-rose-900/40 dark:text-rose-300">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0zM12 9v4M12 17h.01" />
            </svg>
          </div>
          <h2 className="text-lg font-bold text-slate-900 dark:text-white">Bir şeyler ters gitti</h2>
          <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">{this.state.msg}</p>
          <button onClick={() => window.location.reload()} className="btn-primary mt-5">
            Sayfayı yenile
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
