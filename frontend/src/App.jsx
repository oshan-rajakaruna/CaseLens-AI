import AppHeader from "./components/AppHeader.jsx";
import HomePage from "./pages/HomePage.jsx";

export default function App() {
  return (
    <div className="app-shell">
      <AppHeader />
      <main>
        <HomePage />
      </main>
    </div>
  );
}
