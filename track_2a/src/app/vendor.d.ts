declare module "*.bin" {
  const path: string;
  export default path;
}

declare module "*.dylib" {
  const path: string;
  export default path;
}

declare module "*.so" {
  const path: string;
  export default path;
}
