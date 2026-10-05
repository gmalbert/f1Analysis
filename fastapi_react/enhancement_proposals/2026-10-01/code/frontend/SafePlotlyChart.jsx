import {useEffect,useRef,useState} from 'react';

export function SafePlotlyChart({node}) {
  const ref = useRef(null);
  const [error,setError] = useState(null);
  useEffect(() => {
    const element = ref.current;
    let chart, observer, disposed = false;
    setError(null);
    import('plotly.js-dist-min').then(async ({default:plotly}) => {
      if (disposed) return;
      chart = plotly;
      await chart.newPlot(element,node.spec.data,{...node.spec.layout,autosize:true},{responsive:true});
      if (disposed) {chart.purge(element);return;}
      observer = new ResizeObserver(() => chart.Plots.resize(element).catch(() => {}));
      observer.observe(element);
    }).catch(err => {if (!disposed) setError(err.message);});
    return () => {disposed=true;observer?.disconnect();if(chart)chart.purge(element);};
  }, [node.spec]);
  return <div className="view-chart" ref={ref} role="img" aria-label={node.label || 'Interactive Plotly chart'}>
    {error && <div role="alert">Chart unavailable: {error}. Other analysis remains available.</div>}
  </div>;
}
