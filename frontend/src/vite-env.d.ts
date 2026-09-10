/// <reference types="vite/client" />

declare module "*.png" {
  const src: string;
  export default src;
}
declare module "*.jpg" {
  const src: string;
  export default src;
}
declare module "*.svg" {
  const src: string;
  export default src;
}

declare module "react-katex" {
  import { ComponentType } from "react";
  interface KatexProps {
    math: string;
    errorColor?: string;
    renderError?: (error: Error) => JSX.Element;
  }
  export const InlineMath: ComponentType<KatexProps>;
  export const BlockMath: ComponentType<KatexProps>;
}
