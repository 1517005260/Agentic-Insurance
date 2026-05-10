import type { Config } from "tailwindcss";

/**
 * Theme A — 深绿米白 (Bloomberg / 金融出版风).
 *
 * 全站强制：禁用 blue / violet / indigo / purple 等 "AI 色"。
 * primary  = 墨绿  (品牌主色，用于 CTA / 高亮)
 * surface  = 象牙  (页面背景)
 * ink      = 墨黑  (主文本)
 * accent   = 古铜  (链接 / 数值高亮 / 引用 sup)
 * 状态色（success/warning/danger/info）用克制的低饱和度变体，避免撞主题。
 */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // 主品牌
        primary: {
          DEFAULT: "#1B4332",
          50:  "#F1F6F2",
          100: "#DCEAE0",
          200: "#B7D3BF",
          300: "#8AB698",
          400: "#5A9772",
          500: "#2F7A52",
          600: "#1B4332", // base
          700: "#15362A",
          800: "#0F2820",
          900: "#091A14",
        },
        accent: {
          DEFAULT: "#A47148",
          50:  "#FAF4ED",
          100: "#F1E2D0",
          200: "#E0C3A1",
          300: "#CFA47A",
          400: "#BB8A5E",
          500: "#A47148",
          600: "#8B5C36",
          700: "#6E4828",
          800: "#52341C",
          900: "#372110",
        },
        // 中性
        surface: {
          DEFAULT: "#F7F4EE",
          raised: "#FFFFFF",
          sunk:   "#EFEAE0",
        },
        ink: {
          DEFAULT: "#1A1A1A",
          muted:   "#5C5A55",
          subtle:  "#8A877F",
          line:    "#D9D4C7",
        },
        // 状态（克制饱和）。
        // success 比 primary-500 更深、更靠墨绿，避免跟 CTA 撞色；
        // info 走中性石板灰，去掉任何蓝感。
        success: { DEFAULT: "#2A6A45", soft: "#DCEAE0" },
        warning: { DEFAULT: "#B07320", soft: "#F4E3C2" },
        danger:  { DEFAULT: "#9C2A2A", soft: "#F1D4D4" },
        info:    { DEFAULT: "#5C5852", soft: "#E5E2DD" },
      },
      fontFamily: {
        sans: [
          "Inter",
          "'Noto Sans SC'",
          "-apple-system",
          "BlinkMacSystemFont",
          "'PingFang SC'",
          "'Segoe UI'",
          "system-ui",
          "sans-serif",
        ],
        serif: [
          "'Source Serif 4'",
          "'Source Serif Pro'",
          "'Noto Sans SC'",
          "Georgia",
          "serif",
        ],
        mono: [
          "'JetBrains Mono'",
          "ui-monospace",
          "Consolas",
          "monospace",
        ],
      },
      borderRadius: {
        DEFAULT: "6px",
        sm: "4px",
        md: "8px",
        lg: "12px",
      },
      boxShadow: {
        card: "0 1px 2px rgba(20, 30, 22, 0.04), 0 1px 6px rgba(20, 30, 22, 0.05)",
        pop:  "0 4px 16px rgba(20, 30, 22, 0.10), 0 1px 4px rgba(20, 30, 22, 0.06)",
        ring: "0 0 0 3px rgba(27, 67, 50, 0.18)",
      },
      keyframes: {
        "fade-in":  { "0%": { opacity: "0" }, "100%": { opacity: "1" } },
        "slide-in-r": {
          "0%": { transform: "translateX(8px)", opacity: "0" },
          "100%": { transform: "translateX(0)", opacity: "1" },
        },
      },
      animation: {
        "fade-in":  "fade-in 120ms ease-out",
        "slide-in-r": "slide-in-r 160ms ease-out",
      },
    },
  },
  plugins: [],
} satisfies Config;
