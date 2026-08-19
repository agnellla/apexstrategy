/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export: the whole point of D9 is that a precomputed scenario grid needs no
  // backend, so there is nothing to cold-start and nothing to pay for.
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
};

export default nextConfig;
