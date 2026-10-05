import {useEffect,useState} from 'react';
export function TabScroll({target}) {
  const [edges,setEdges]=useState({left:false,right:false});
  useEffect(()=>{
    const el=target.current;if(!el)return;
    const update=()=>setEdges({left:el.scrollLeft>1,right:el.scrollLeft+el.clientWidth<el.scrollWidth-1});
    const resize=new ResizeObserver(update);resize.observe(el);el.addEventListener('scroll',update);update();
    return()=>{resize.disconnect();el.removeEventListener('scroll',update);};
  },[target]);
  return <>{edges.left && <button className="tab-scroll left" aria-label="Scroll tabs left" onClick={()=>target.current?.scrollBy({left:-240,behavior:'smooth'})}>‹</button>}{edges.right && <button className="tab-scroll right" aria-label="Scroll tabs right" onClick={()=>target.current?.scrollBy({left:240,behavior:'smooth'})}>›</button>}</>;
}
