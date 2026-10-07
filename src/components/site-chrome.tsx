import { Link } from '@tanstack/react-router';
import { Eye, ArrowUpRight } from 'lucide-react';

export function SiteHeader() {
  return <header className="site-header"><Link to="/" className="brand"><span className="brand-mark"><Eye aria-hidden="true" /></span>FactLens</Link><nav aria-label="Main navigation" className="site-nav"><Link to="/detector">Detector</Link><Link to="/methodology">Methodology</Link><a className="source-link" href="https://github.com/suban7048-source/FactLens-web" target="_blank" rel="noreferrer">Source <ArrowUpRight className="inline size-3" /></a></nav></header>;
}
export function SiteFooter() {
  return <footer className="site-footer"><span>FactLens © 2026</span><span>A little more context. A little less noise.</span></footer>;
}