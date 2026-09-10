import { Suspense, lazy } from "react";
import { Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import ScrollToTop from "./components/ScrollToTop";
import ErrorBoundary from "./components/ErrorBoundary";
import ConnectionBanner from "./components/ConnectionBanner";
import Spinner from "./components/Spinner";
import Overview from "./pages/Overview";

// ağır sayfalar (recharts/leaflet) tembel yüklenir → küçük ilk paket
const Predictor = lazy(() => import("./pages/Predictor"));
const CriticalPGA = lazy(() => import("./pages/CriticalPGA"));
const MapView = lazy(() => import("./pages/MapView"));
const Methodology = lazy(() => import("./pages/Methodology"));
const Validation = lazy(() => import("./pages/Validation"));
const Results = lazy(() => import("./pages/Results"));

function PageFallback() {
  return (
    <div className="flex min-h-[50vh] items-center justify-center">
      <Spinner label="Sayfa yükleniyor…" />
    </div>
  );
}

export default function App() {
  return (
    <div className="min-h-screen">
      <ConnectionBanner />
      <Navbar />
      <ScrollToTop />
      <main className="mx-auto max-w-7xl px-4 py-8">
        <ErrorBoundary>
          <Suspense fallback={<PageFallback />}>
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/predictor" element={<Predictor />} />
              <Route path="/critical-pga" element={<CriticalPGA />} />
              <Route path="/map" element={<MapView />} />
              <Route path="/methodology" element={<Methodology />} />
              <Route path="/validation" element={<Validation />} />
              <Route path="/results" element={<Results />} />
            </Routes>
          </Suspense>
        </ErrorBoundary>
      </main>
    </div>
  );
}
