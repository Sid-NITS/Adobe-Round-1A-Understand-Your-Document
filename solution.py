import fitz  # PyMuPDF
import cv2
import json
import re
from collections import defaultdict, Counter

# Wrap numpy import to detect numpy2 compatibility issues
try:
    import numpy as np
except ImportError:
    raise ImportError("NumPy import failed. Please ensure numpy<2 is installed for compatibility.")

# Attempt to import pytesseract, catching pandas/numpy2 errors
try:
    import pytesseract
except ImportError as e:
    raise ImportError(
        "pytesseract (and its pandas dependency) was compiled against NumPy<2. "
        "Please install numpy<2 (e.g. 'pip install 'numpy<2') to proceed."    
    ) from e

# ------------------------------
# CONFIGURABLE WEIGHTS & PARAMS
# ------------------------------
WEIGHTS_TITLE = {
    'numbering':    5,
    'font_size':    3,
    'bold':         2,
    'all_caps':     2,
    'centered':     2,
    'spacing':      1,
    'body_penalty': -5
}
BOX_EXPANSION = 2.0
DPI = 300
MIN_CONF = 50

# Heading extraction configs
SCORE_THRESHOLD = 4
START_PAGE = 1  # skip first page (title)
WEIGHTS_HEAD = WEIGHTS_TITLE
NUMBERING_PATTERNS = [
    re.compile(r"^\d+\."),
    re.compile(r"^\d+\.\d+\."),
    re.compile(r"^Chapter \d+", re.I),
    re.compile(r"^Appendix [A-Z]", re.I)
]

# --- Title Extraction ---
def extract_pdf_title(pdf_path, weights=WEIGHTS_TITLE, box_factor=BOX_EXPANSION, dpi=DPI):
    doc = fitz.open(pdf_path)
    if not doc:
        raise RuntimeError("Error opening PDF or PDF is empty.")
    page = doc.load_page(0)
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    data = page.get_text("dict")
    spans = []
    for block in data.get('blocks', []):
        if block.get('type') != 0: continue
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                text = span.get('text', '').strip()
                if not text: continue
                spans.append({'text': text, 'bbox': span['bbox'], 'size': span['size'], 'flags': span['flags']})
    sizes = [s['size'] for s in spans]
    body_med = float(np.median(sizes)) if sizes else 12.0
    scored = []
    spans_sorted = sorted(spans, key=lambda s: s['bbox'][1])
    y_prev = spans_sorted[0]['bbox'][1] if spans_sorted else 0
    page_w = page.rect.width
    for s in spans_sorted:
        score = 0
        txt = s['text']
        x0,y0,x1,y1 = s['bbox']
        if txt[0].isdigit() and '.' in txt.split()[0]: score += weights['numbering']
        score += weights['font_size'] * max(0, s['size'] - body_med)
        if s['flags'] & (1<<4): score += weights['bold']
        if txt.isupper() and len(txt)>3: score += weights['all_caps']
        if abs((x0+x1)/2 - page_w/2) < page_w*0.15: score += weights['centered']
        if (y0-y_prev) > s['size']*1.5: score += weights['spacing']
        if abs(s['size']-body_med)<0.5: score += weights['body_penalty']
        scored.append((score, s))
        y_prev = y1
    best_score, best = max(scored, key=lambda x: x[0])
    scale = dpi/72.0
    x0,y0,x1,y1 = [c*scale for c in best['bbox']]
    w,h = x1-x0, y1-y0
    cx,cy = (x0+x1)/2, (y0+y1)/2
    nw,nh = w*box_factor, h*box_factor
    x0c = max(0, int(cx-nw/2)); y0c = max(0, int(cy-nh/2))
    x1c = min(img.shape[1], int(cx+nw/2)); y1c = min(img.shape[0], int(cy+nh/2))
    crop = img[y0c:y1c, x0c:x1c]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, proc = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    ocr = pytesseract.image_to_data(proc, output_type=pytesseract.Output.DICT, config='--oem 1 --psm 3')
    words = [ocr['text'][i].strip() for i, c in enumerate(ocr['conf']) 
             if isinstance(c, (int, float)) and c > MIN_CONF and ocr['text'][i].strip()]
    title_text = " ".join(words)
    return title_text

# --- Heading Extraction ---
def extract_toc(doc):
    toc = doc.get_toc()
    return [{'level':f"H{item[0]}", 'text':item[1], 'page':item[2]-1} for item in toc] if toc else []

def extract_blocks(doc):
    spans=[]; sid=0
    for pi in range(START_PAGE, len(doc)):
        page = doc.load_page(pi)
        for b in page.get_text('dict')['blocks']:
            if 'lines' not in b: continue
            for line in b['lines']:
                for span in line['spans']:
                    t=span['text'].strip()
                    if not t: continue
                    spans.append({
                        'id':sid,'text':t,'page':pi+1,'font_size':span['size'],
                        'bold':'Bold' in span['font'],'x0':span['bbox'][0],
                        'x1':span['bbox'][2],'y0':span['bbox'][1],'y1':span['bbox'][3],
                        'spacing_above':None
                    })
                    sid+=1
    spans_sorted=sorted(spans,key=lambda s:(s['page'],s['y0'],s['x0']))
    for i in range(1,len(spans_sorted)):
        prev=spans_sorted[i-1]; curr=spans_sorted[i]
        if curr['page']==prev['page']:
            curr['spacing_above'] = max(0, curr['y0']-prev['y1'])
        else:
            curr['spacing_above'] = 0
    return spans_sorted

def compute_body_baseline_sizes(spans, threshold=0.8):
    sizes=[round(s['font_size']) for s in spans]
    cnt=Counter(sizes)
    total=len(sizes)
    if total==0: return {10}
    sorted_sz=sorted(cnt.items(), key=lambda x:x[1], reverse=True)
    base=set(); cum=0
    for sz,c in sorted_sz:
        base.add(sz); cum+=c
        if cum/total>=threshold: break
    main=sorted_sz[0][0]
    for sz,_ in sorted_sz:
        if abs(sz-main)<=1: base.add(sz)
    return base

def score_span(span, baseline, page_w, med_sp):
    score=0; t=span['text']
    maxb=max(baseline) if baseline else 10
    if any(p.match(t) for p in NUMBERING_PATTERNS): score+=WEIGHTS_HEAD['numbering']
    if span['font_size']>=maxb+1.5: score+=WEIGHTS_HEAD['font_size']
    if span['bold']: score+=WEIGHTS_HEAD['bold']
    if t.isupper() and len(t)>5: score+=WEIGHTS_HEAD['all_caps']
    w=span['x1']-span['x0']
    if abs((page_w-w)/2-span['x0'])<10: score+=WEIGHTS_HEAD['centered']
    # use safe spacing check
    sp_val = span.get('spacing_above', 0) or 0
    if sp_val>med_sp.get(span['page'],0)*1.5: score+=WEIGHTS_HEAD['spacing']
    if round(span['font_size']) in baseline: score+=WEIGHTS_HEAD['body_penalty']
    return score

def merge_colinear_text(cands, groups):
    merged=[]; seen=set()
    for c in sorted(cands, key=lambda s:(s['page'],s['y0'],s['x0'])):
        if c['id'] in seen: continue
        line_group=[c]
        for s in groups[c['page']]:
            if s['id']==c['id']: continue
            if max(c['y0'],s['y0'])<min(c['y1'],s['y1']) and abs(c['font_size']-s['font_size'])<0.1:
                line_group.append(s)
        line_group.sort(key=lambda s:s['x0'])
        merged.append({
            'id':c['id'],'text':' '.join(s['text'] for s in line_group),'font_size':c['font_size'],
            'page':c['page'],'x0':min(s['x0'] for s in line_group),'x1':max(s['x1'] for s in line_group),
            'y0':c['y0'],'y1':c['y1']
        })
        for s in line_group: seen.add(s['id'])
    return merged

def merge_multiline(cands):
    merged=[]; i=0
    while i<len(cands):
        cur=cands[i]; j=i+1
        while j<len(cands) and cands[j]['page']==cur['page'] and abs(cur['font_size']-cands[j]['font_size'])<0.1 \
              and (cands[j]['y0']-cur['y1'])<cur['font_size']*1.2:
            cur['text']+=' '+cands[j]['text']; cur['y1']=cands[j]['y1']; j+=1
        merged.append(cur); i=j
    return merged

def assign_heading_levels(cands):
    sizes=sorted({c['font_size'] for c in cands}, reverse=True)
    lvl={sz:f"H{i+1}" for i,sz in enumerate(sizes)}
    return [{
        'level':lvl[c['font_size']],'text':c['text'],'page':c['page']-1
    } for c in cands]

def extract_headings_from_pdf(pdf_path):
    doc=fitz.open(pdf_path)
    toc=extract_toc(doc)
    if toc: return toc
    spans=extract_blocks(doc)
    baseline=compute_body_baseline_sizes(spans)
    page_w=doc.load_page(START_PAGE).rect.width
    # safe median spacing
    med_sp={}
    pages={s['page'] for s in spans}
    for p in pages:
        vals=[s['spacing_above'] for s in spans if s['page']==p and isinstance(s['spacing_above'],(int,float)) and s['spacing_above']>0]
        med_sp[p]=float(np.median(vals)) if vals else 0
    for s in spans: s['score']=score_span(s,baseline,page_w,med_sp)
    cands=[s for s in spans if s['score']>=SCORE_THRESHOLD]
    groups=defaultdict(list)
    for s in spans: groups[s['page']].append(s)
    merged=merge_colinear_text(cands,groups)
    merged=merge_multiline(sorted(merged, key=lambda s:(s['page'],s['y0'])))
    return assign_heading_levels(merged)

if __name__=='__main__':
    PDF_PATH = './input/file02.pdf'
    try:
        title = extract_pdf_title(PDF_PATH)
        outline = extract_headings_from_pdf(PDF_PATH)
        result = {'title': title, 'outline': outline}
        with open('result.json','w',encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False,indent=2)
        print(f"✅ Saved extraction result to 'outline.json' ({len(outline)} headings)")
    except Exception as e:
        print(f"❌ Error: {e}")
