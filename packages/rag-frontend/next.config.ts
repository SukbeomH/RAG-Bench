import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  webpack: (config) => {
    // pdf.js worker support
    config.resolve.alias.canvas = false;
    return config;
  },
};

export default nextConfig;
