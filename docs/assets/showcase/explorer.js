import { checkboxField, element, log, selectField, sourceLink, startShowcase } from './ui.js';

function matchingLabels(data,query) {
  const needle=query.trim().toLocaleLowerCase();
  return data.labels.filter(label=>`${label.code} ${label.title}`.toLocaleLowerCase().includes(needle)||data.samples.some(sample=>sample.label===label.code&&`${sample.id} ${sample.text}`.toLocaleLowerCase().includes(needle)));
}

function renderCorpus(root,data) {
  const controls=root.querySelector('.sc-controls');
  const stage=root.querySelector('.sc-stage');
  const details=root.querySelector('.sc-details');
  const live=root.querySelector('[role="status"]');
  const first=data.labels.find(label=>label.code===data.samples[0].label);
  const state={label:first.code,sample:0,reveal:true,query:''};
  const searchLabel=element('label','sc-field');
  searchLabel.append(element('span','','Find a subject or phrase'));
  const search=element('input','sc-search');search.type='search';search.setAttribute('aria-label','Find a subject or phrase');search.placeholder='Code, subject, record ID, or source text';searchLabel.append(search);
  const sampleField=selectField('Source abstract',[],'abstract');
  const reveal=checkboxField('Show corpus label','reveal');
  controls.replaceChildren(searchLabel,sampleField.wrapper,reveal.wrapper);
  const layout=element('div','sc-corpus-layout');
  const subjects=element('div','sc-subjects');subjects.setAttribute('aria-label','GRNTI subjects');
  const record=element('article','sc-record');
  layout.append(subjects,record);stage.replaceChildren(layout);

  function showRecord(reset=false) {
    const label=data.labels.find(item=>item.code===state.label);
    const samples=data.samples.filter(sample=>sample.label===state.label);
    if(reset){state.sample=0;sampleField.select.replaceChildren(...samples.map((sample,index)=>{const option=element('option','',`Record ${sample.id}`);option.value=String(index);return option;}));}
    sampleField.select.disabled=samples.length===0;
    record.replaceChildren(element('h3','',`${label.code} · ${label.title}`));
    if(samples.length){const sample=samples[state.sample];record.append(element('p','sc-text',sample.text));if(state.reveal)record.append(element('p','sc-label',`Corpus label: ${sample.label} · ${sample.label_text}`));record.append(element('p','sc-live',`Source record ${sample.id}. Original text is preserved.`));}
    else record.append(element('p','','The published label catalogue includes this subject, but the committed smoke corpus has no abstract for it.'));
    details.replaceChildren(element('h3','','A labelled corpus'),element('p','',`${data.labels.length} model subjects. ${data.samples.length} committed abstracts.`),element('p','','The displayed label is supplied by the dataset. It is not a new model result.'),sourceLink(data.source,'Read the source records'),sourceLink(data.label_source,'Pinned model label catalogue'));
    log('corpus:record',{code:label.code,samples:samples.length,reveal:state.reveal});
  }
  function showSubjects() {
    const labels=matchingLabels(data,state.query);
    subjects.replaceChildren(...labels.map(label=>{
      const button=element('button','sc-button',`${label.code} · ${label.title}`);button.type='button';button.setAttribute('aria-pressed',String(label.code===state.label));
      button.addEventListener('click',()=>{state.label=label.code;showSubjects();showRecord(true);});
      return button;
    }));
    live.textContent=`${labels.length} of ${data.labels.length} subjects match. The smoke corpus covers 20 distinct subject codes.`;
  }
  search.addEventListener('input',()=>{state.query=search.value;showSubjects();});
  sampleField.select.addEventListener('change',()=>{state.sample=Number(sampleField.select.value);showRecord();});
  reveal.input.addEventListener('change',()=>{state.reveal=reveal.input.checked;showRecord();});
  showSubjects();showRecord(true);
}

startShowcase('corpus',renderCorpus,new URL('./data.json',import.meta.url));
