import {useEffect,useState} from 'react';

export function useTheme() {
  const [theme,setTheme]=useState(()=>document.documentElement.dataset.theme || 'light');
  useEffect(()=>{
    const update=()=>setTheme(document.documentElement.dataset.theme || 'light');
    const observer=new MutationObserver(update);
    observer.observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
    update();
    return()=>observer.disconnect();
  },[]);
  return theme;
}
