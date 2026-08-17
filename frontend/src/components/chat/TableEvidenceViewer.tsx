import {useEffect,useMemo,useState} from 'react';
import {api} from '../../api/corporateBrain';

function columnNumber(label:string){return label.split('').reduce((value,char)=>value*26+char.charCodeAt(0)-64,0)}
function rangeBounds(range:string|null,row:number|null,rowEnd:number|null){
  const match=range?.match(/^([A-Z]+)(\d+):([A-Z]+)(\d+)$/i);
  if(match)return {r1:+match[2],r2:+match[4],c1:columnNumber(match[1].toUpperCase()),c2:columnNumber(match[3].toUpperCase())};
  return {r1:row??0,r2:rowEnd??row??0,c1:1,c2:Number.MAX_SAFE_INTEGER};
}

export function TableEvidenceViewer({fileHash,sheet,cellRange,row,rowEnd}:{fileHash:string;sheet:string|null;cellRange:string|null;row:number|null;rowEnd:number|null}){
  const [data,setData]=useState<{kind:string;sheet:string|null;sheets:string[];rows:unknown[][]}|null>(null);
  const [error,setError]=useState('');
  useEffect(()=>{api.tableEvidence(fileHash,sheet??undefined).then(setData).catch(e=>setError(e instanceof Error?e.message:'Could not open table evidence.'))},[fileHash,sheet]);
  const bounds=useMemo(()=>rangeBounds(cellRange,row,rowEnd),[cellRange,row,rowEnd]);
  if(error)return <p className="error">{error}</p>;
  if(!data)return <p className="table-viewer__loading">Loading exact table evidence…</p>;
  const start=Math.max(0,bounds.r1-4);const end=Math.min(data.rows.length,bounds.r2+3);
  return <div className="table-viewer" role="region" aria-label="Exact table evidence"><div className="table-viewer__meta">{data.kind==='xlsx'?`Sheet: ${data.sheet}`:'CSV rows'} · showing {start+1}–{end}</div><div className="table-viewer__scroll"><table><tbody>{data.rows.slice(start,end).map((values,index)=>{const actualRow=start+index+1;return <tr key={actualRow}><th>{actualRow}</th>{values.map((value,columnIndex)=>{const actualColumn=columnIndex+1;const selected=actualRow>=bounds.r1&&actualRow<=bounds.r2&&actualColumn>=bounds.c1&&actualColumn<=bounds.c2;return <td key={columnIndex} className={selected?'table-viewer__selected':''}>{value==null?'':String(value)}</td>})}</tr>})}</tbody></table></div></div>
}
