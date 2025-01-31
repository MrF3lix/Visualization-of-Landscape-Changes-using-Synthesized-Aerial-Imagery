import "./globals.css";

export const metadata = {
  title: "VT1 Editor"
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
