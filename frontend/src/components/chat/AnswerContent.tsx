import type {ChatResponse} from '../../types';

function pairs(text:string){return [...text.matchAll(/(?:^|\|)\s*([^=|:]+?)\s*(?:=|:)\s*([^|]+)/g)].map(match=>[match[1].trim(),match[2].trim()] as const)}

export function AnswerContent({response,text}:{response?:ChatResponse;text:string}){
  if(!response)return <div>{text}</div>;
  const rows=response.sources.map(source=>Object.fromEntries(pairs(source.text))).filter(row=>Object.keys(row).length>1);
  if((response.result_type==='TABLE_RESULT'||response.result_type==='SECTION_RESULT')&&rows.length){const columns=[...new Set(rows.flatMap(row=>Object.keys(row)))];return <div className="answer-table-wrap"><table className="answer-table"><thead><tr>{columns.map(column=><th key={column}>{column}</th>)}</tr></thead><tbody>{rows.map((row,index)=><tr key={index}>{columns.map(column=><td key={column}>{row[column]??'—'}</td>)}</tr>)}</tbody></table></div>}
  if(response.result_type==='MULTI_VALUE'||response.result_type==='MULTI_MENTION'){const values=text.split('\n').map(value=>value.replace(/^[-•]\s*/, '').trim()).filter(value=>value&&!value.endsWith(':'));return <div className="multi-answer">{values.map((value,index)=><span key={`${value}-${index}`}>{value}</span>)}</div>}
  return <div>{text}</div>
}
