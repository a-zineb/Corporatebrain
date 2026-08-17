import {useEffect,useState} from 'react';
import {ArrowLeft,Download,ExternalLink,X} from 'lucide-react';
import {api} from '../../api/corporateBrain';
import type {Source} from '../../types';
import {TableEvidenceViewer} from './TableEvidenceViewer';

export function SourcePanel({source,onClose}:{source:Source;onClose:()=>void}){
  const original=api.originalUrl(source.file_hash,false);
  const isPdf=source.file_type==='pdf';
  const isWord=source.file_type==='docx'||source.file_type==='doc';
  const [previewPage,setPreviewPage]=useState<number|null>(null);const [previewError,setPreviewError]=useState('');
  useEffect(()=>{if(isWord)api.previewInfo(source.file_hash,source.block_id).then(info=>setPreviewPage(info.page)).catch(e=>setPreviewError(e instanceof Error?e.message:'Preview unavailable.'))},[isWord,source.file_hash,source.block_id]);
  const previewReady=isWord&&Boolean(previewPage);
  const exact=isPdf&&source.page?`${original}#page=${source.page}&zoom=page-width`:previewReady?`${api.previewUrl(source.file_hash)}#page=${previewPage}&zoom=page-width`:original;
  const tabular=source.file_type==='xlsx'||source.file_type==='csv';
  return <div className="source-panel"><header><button onClick={onClose}><ArrowLeft/> Back</button><button onClick={onClose} aria-label="Close"><X/></button></header><span className="badge">{source.file_type.toUpperCase()}</span><h2>{source.document}</h2><small>{source.location}</small>{source.sheet&&<p><strong>Sheet:</strong> {source.sheet}{source.cell_range&&<> · <strong>Cells:</strong> {source.cell_range}</>}</p>}{source.file_type==='csv'&&source.row&&<p><strong>CSV row:</strong> {source.row_end&&source.row_end!==source.row?`${source.row}–${source.row_end}`:source.row}</p>}<div className="source-actions"><a href={exact} target="_blank" rel="noreferrer"><ExternalLink size={15}/> Open exact evidence</a><a href={api.originalUrl(source.file_hash,true)}><Download size={15}/> Open original file</a></div>{(isPdf||previewReady)&&<iframe title={`Source ${source.document} ${source.location}`} src={exact} className="source-document-frame"/>}{isWord&&!previewReady&&<p className={previewError?'error':''}>{previewError||'Generating page-stable preview…'}</p>}{tabular&&<TableEvidenceViewer fileHash={source.file_hash} sheet={source.sheet} cellRange={source.cell_range} row={source.row} rowEnd={source.row_end}/>}<div className="highlight"><strong>Exact evidence</strong><p>{source.text}</p></div></div>
}
