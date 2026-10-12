(function(){
  const $ = (s, r=document) => r.querySelector(s), $$ = (s, r=document) => Array.from(r.querySelectorAll(s));
  const store = {
    get(k, d){ try{ const v = localStorage.getItem('ia2e2:'+k); return v === null ? d : JSON.parse(v); }catch(e){ return d; } },
    set(k, v){ try{ localStorage.setItem('ia2e2:'+k, JSON.stringify(v)); }catch(e){} }
  };

  /* ---------- clases de NObeyesdad (de menor a mayor IMC) ---------- */
  const CLASES = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II',
                  'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'];
  const color = i => 'var(--c'+(i+1)+')';
  const idx = {}; CLASES.forEach((c,i) => idx[c] = i);

  // nombres de clase en el texto -> punto de color
  $$('code').forEach(el => { const t = el.textContent.trim(); if (t in idx){ el.classList.add('cls'); el.style.setProperty('--cc', color(idx[t])); } });

  /* ---------- progreso por sección ---------- */
  const total = $$('.sec').length;
  let hechas = store.get('secs', {});
  function pintarProgreso(){
    let n = 0;
    $$('.done-btn').forEach(b => { const on = !!hechas[b.dataset.sec]; if (on) n++;
      b.setAttribute('aria-pressed', on); $('.lbl', b).textContent = on ? 'Sección estudiada' : 'Marcar sección como estudiada';
      const a = $('a[data-nav="'+b.dataset.sec+'"]'); if (a) a.classList.toggle('is-done', on); });
    $('#prog-bar').style.width = (n/total*100)+'%';
    $('#prog-txt').textContent = n+' de '+total+' secciones estudiadas';
    $('#top-prog').textContent = n+'/'+total;
  }
  $$('.done-btn').forEach(b => b.addEventListener('click', () => { hechas[b.dataset.sec] = !hechas[b.dataset.sec]; store.set('secs', hechas); pintarProgreso(); }));
  pintarProgreso();

  /* ---------- navegación: sección actual + menú móvil ---------- */
  const links = {}; $$('.side nav a').forEach(a => links[a.dataset.nav] = a);
  if ('IntersectionObserver' in window){
    const obs = new IntersectionObserver(es => { es.forEach(e => { if (e.isIntersecting){
      Object.values(links).forEach(a => a.classList.remove('is-current'));
      const a = links[e.target.id]; if (a){ a.classList.add('is-current'); } } }); }, {rootMargin:'-35% 0px -60% 0px'});
    $$('.sec').forEach(s => obs.observe(s));
  }
  const openBtn = $('#nav-open'); let scrim = null;
  function cerrarNav(){ document.body.classList.remove('nav-open'); openBtn.setAttribute('aria-expanded','false'); if (scrim){ scrim.remove(); scrim = null; } }
  openBtn.addEventListener('click', () => { document.body.classList.add('nav-open'); openBtn.setAttribute('aria-expanded','true');
    scrim = document.createElement('div'); scrim.className = 'scrim'; scrim.addEventListener('click', cerrarNav); document.body.appendChild(scrim); });
  $$('.side nav a').forEach(a => a.addEventListener('click', cerrarNav));

  /* ---------- pasos plegables ---------- */
  function plegar(step, on){ step.classList.toggle('is-collapsed', on); const b = $('.step-toggle', step); b.textContent = on ? 'Expandir' : 'Contraer'; b.setAttribute('aria-expanded', !on); }
  $$('.step').forEach(s => $('.step-toggle', s).addEventListener('click', () => plegar(s, !s.classList.contains('is-collapsed'))));
  $$('.js-expand-all').forEach(b => b.addEventListener('click', () => {
    const sec = b.closest('.sec'), steps = $$('.step', sec), cerrar = !steps.every(s => s.classList.contains('is-collapsed'));
    steps.forEach(s => plegar(s, cerrar)); b.textContent = cerrar ? 'Expandir todos los pasos' : 'Contraer todos los pasos'; }));
  // si se salta a un paso plegado, abrirlo
  $$('a[href^="#"]').forEach(a => a.addEventListener('click', () => { const t = document.getElementById(a.getAttribute('href').slice(1)); if (t && t.classList.contains('step')) plegar(t, false); }));

  /* ---------- filtro de tablas largas ---------- */
  $$('.table-wrap').forEach((wrap, k) => {
    const filas = $$('tbody tr', wrap); if (filas.length < 10) return;
    const box = document.createElement('div'); box.className = 'tfilter';
    box.innerHTML = '<label for="tf'+k+'">Filtrar filas</label><input id="tf'+k+'" type="search" placeholder="escribe para filtrar…" autocomplete="off"><span></span>';
    wrap.before(box);
    const inp = $('input', box), out = $('span', box);
    inp.addEventListener('input', () => { const q = inp.value.trim().toLowerCase(); let n = 0;
      filas.forEach(r => { const ok = !q || r.textContent.toLowerCase().includes(q); r.hidden = !ok; if (ok) n++; });
      out.textContent = q ? n+' de '+filas.length : ''; });
  });

  /* ---------- buscador de teoría ---------- */
  const qt = $('#q-teoria');
  if (qt){
    const sec = $('#teoria'), grupos = $$('details.group', sec);
    const items = grupos.flatMap(g => $$('.prose > ul > li, .prose > ol > li, .prose > p', g));
    items.forEach(it => it.dataset.orig = it.innerHTML);
    qt.addEventListener('input', () => {
      const q = qt.value.trim().toLowerCase(); let n = 0;
      items.forEach(it => { it.innerHTML = it.dataset.orig; const ok = !q || it.textContent.toLowerCase().includes(q); it.hidden = !ok; if (ok && q){ n++; resaltar(it, q); } });
      grupos.forEach(g => { const vis = $$('.prose > *', g).some(e => !e.hidden && (e.tagName !== 'UL' && e.tagName !== 'OL' || $$(':scope > li', e).some(li => !li.hidden)));
        const lists = $$('.prose > ul, .prose > ol', g); lists.forEach(l => l.hidden = !$$(':scope > li', l).some(li => !li.hidden));
        g.hidden = !!q && !vis; if (q) g.open = true; });
      $('#q-teoria-n').textContent = q ? (n ? n+(n === 1 ? ' coincidencia' : ' coincidencias') : 'Sin coincidencias. Prueba con otra palabra.') : '';
    });
  }
  function resaltar(el, q){
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT); const nodos = [];
    while (walker.nextNode()) nodos.push(walker.currentNode);
    nodos.forEach(t => { const i = t.nodeValue.toLowerCase().indexOf(q); if (i < 0) return;
      const r = document.createRange(); r.setStart(t, i); r.setEnd(t, i + q.length); const m = document.createElement('mark'); r.surroundContents(m); });
  }

  /* ---------- copiar código ---------- */
  $$('pre').forEach(pre => { const b = document.createElement('button'); b.type = 'button'; b.className = 'copy'; b.textContent = 'Copiar';
    b.addEventListener('click', () => { const txt = $('code', pre) ? $('code', pre).textContent : pre.textContent;
      const ok = () => { b.textContent = 'Copiado'; setTimeout(() => b.textContent = 'Copiar', 1400); };
      if (navigator.clipboard && navigator.clipboard.writeText){ navigator.clipboard.writeText(txt).then(ok, () => seleccionar(pre)); } else seleccionar(pre); });
    pre.appendChild(b); });
  function seleccionar(pre){ const r = document.createRange(); r.selectNodeContents($('code', pre) || pre); const s = getSelection(); s.removeAllRanges(); s.addRange(r); }

  /* ---------- lightbox ---------- */
  const lb = $('#lb'), lbImg = $('#lb-img'); let lista = [], pos = 0, ultimo = null;
  function abrir(btn){
    const g = btn.dataset.group;
    lista = $$('.fig-btn[data-group="'+g+'"]').filter(b => !b.closest('[hidden]'));
    pos = Math.max(0, lista.indexOf(btn)); ultimo = btn; lb.hidden = false; lb.classList.remove('zoom'); document.body.style.overflow = 'hidden'; mostrar(); $('#lb-close').focus();
  }
  function mostrar(){ const b = lista[pos]; lbImg.src = b.dataset.src; lbImg.alt = b.dataset.cap; $('#lb-cap').textContent = b.dataset.cap;
    $('#lb-pos').textContent = (pos+1)+' / '+lista.length; $('#lb-prev').disabled = $('#lb-next').disabled = lista.length < 2; }
  function cerrar(){ lb.hidden = true; document.body.style.overflow = ''; lbImg.src = ''; if (ultimo) ultimo.focus(); }
  function mover(d){ if (lista.length < 2) return; pos = (pos + d + lista.length) % lista.length; lb.classList.remove('zoom'); mostrar(); }
  document.addEventListener('click', e => { const b = e.target.closest('.fig-btn'); if (b) abrir(b); });
  $('#lb-close').addEventListener('click', cerrar); $('#lb-prev').addEventListener('click', () => mover(-1)); $('#lb-next').addEventListener('click', () => mover(1));
  function zoom(){ lb.classList.toggle('zoom'); $('#lb-zoom').textContent = lb.classList.contains('zoom') ? 'Ajustar a pantalla' : 'Tamaño real'; }
  $('#lb-zoom').addEventListener('click', zoom); lbImg.addEventListener('click', zoom);
  $('#lb-stage').addEventListener('click', e => { if (e.target.id === 'lb-stage') cerrar(); });
  document.addEventListener('keydown', e => { if (lb.hidden) return; if (e.key === 'Escape') cerrar(); if (e.key === 'ArrowRight') mover(1); if (e.key === 'ArrowLeft') mover(-1); });

  /* ---------- galería ---------- */
  let galEtapa = 'todas', galTipo = 'grafica';
  function filtrarGaleria(){ $$('.gal-item').forEach(it => { it.hidden = !((galEtapa === 'todas' || it.dataset.etapa === galEtapa) && (galTipo === 'todo' || it.dataset.tipo === galTipo)); }); }
  $$('#galeria .fchip').forEach(c => c.addEventListener('click', () => { $$('#galeria .fchip').forEach(x => x.classList.remove('is-on')); c.classList.add('is-on'); galEtapa = c.dataset.etapa; filtrarGaleria(); }));
  $$('#galeria .seg button').forEach(c => c.addEventListener('click', () => { $$('#galeria .seg button').forEach(x => x.classList.remove('is-on')); c.classList.add('is-on'); galTipo = c.dataset.tipo; filtrarGaleria(); }));
  filtrarGaleria();

  /* ---------- preguntas ---------- */
  const cards = $$('.qcard'); let dominadas = store.get('qs', {}), catActual = 'todas';
  function revelar(card, on){ const a = $('.qans', card), b = $('.qreveal', card); a.hidden = !on; b.textContent = on ? 'Ocultar respuesta' : 'Ver respuesta'; b.setAttribute('aria-expanded', on); }
  cards.forEach(c => { $('.qreveal', c).addEventListener('click', () => revelar(c, $('.qans', c).hidden));
    const chk = $('input', c); chk.checked = !!dominadas[c.dataset.n]; c.classList.toggle('is-done', chk.checked);
    chk.addEventListener('change', () => { dominadas[c.dataset.n] = chk.checked; store.set('qs', dominadas); c.classList.toggle('is-done', chk.checked); pintarQ(); filtrarQ(); }); });
  function pintarQ(){ const n = cards.filter(c => $('input', c).checked).length; const bar = $('#q-bar'); if (!bar) return;
    bar.style.width = (n/cards.length*100)+'%'; $('#q-count').textContent = n+' de '+cards.length+' dominadas'; }
  function filtrarQ(){ const ocultar = $('#q-hide-done') && $('#q-hide-done').checked;
    cards.forEach(c => c.hidden = !((catActual === 'todas' || c.dataset.cat === catActual) && !(ocultar && $('input', c).checked))); }
  $$('#preguntas .fchip').forEach(ch => ch.addEventListener('click', () => { $$('#preguntas .fchip').forEach(x => x.classList.remove('is-on')); ch.classList.add('is-on'); catActual = ch.dataset.cat; filtrarQ(); }));
  if ($('#q-hide-done')) $('#q-hide-done').addEventListener('change', filtrarQ);
  if ($('#q-all')) $('#q-all').addEventListener('click', e => { const abrirTodas = cards.some(c => $('.qans', c).hidden); cards.forEach(c => revelar(c, abrirTodas)); e.target.textContent = abrirTodas ? 'Ocultar todas las respuestas' : 'Mostrar todas las respuestas'; });
  if ($('#q-random')) $('#q-random').addEventListener('click', () => {
    let pool = cards.filter(c => !c.hidden && !$('input', c).checked); if (!pool.length) pool = cards.filter(c => !c.hidden); if (!pool.length) return;
    const c = pool[Math.floor(Math.random()*pool.length)]; cards.forEach(x => x.classList.remove('flash')); revelar(c, false);
    c.classList.add('flash'); c.scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block:'center'}); $('.qreveal', c).focus({preventScroll:true}); });
  pintarQ(); filtrarQ();
})();
