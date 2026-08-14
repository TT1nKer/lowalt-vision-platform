const $ = (selector) => document.querySelector(selector);
const formatNumber = (value) => Number(value || 0).toLocaleString('zh-CN');

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || `HTTP ${response.status}`);
  return response.json();
}

const supportPresentation = {
  vehicle_row_supported: { label: '车排支持', color: '#15a66a', className: 'high' },
  vehicle_detected: { label: '检测到车辆', color: '#e09a27', className: 'medium' },
  segformer_only: { label: '仅模型候选', color: '#718096', className: 'low' },
};

async function loadEngineering() {
  const summary = await getJson('/api/platform/summary');
  $('#engImagery').textContent = formatNumber(summary.imagery.count);
  $('#engCandidates').textContent = formatNumber(summary.parking_candidates.total);
  $('#engSupported').textContent = formatNumber(summary.parking_candidates.vehicle_row_supported);
  const progress = $('#secondaryProgress');
  const completed = summary.secondary_analysis.completed;
  const total = summary.secondary_analysis.total;
  progress.textContent = `${formatNumber(completed)} / ${formatNumber(total)}`;
  progress.className = `state ${completed === total ? 'ready' : 'pending'}`;
  bindImportControls();
  loadImportRuns();
}

const importStatusPresentation = {
  pending: ['等待中', 'pending'],
  running: ['导入中', 'pending'],
  done: ['已完成', 'ready'],
  error: ['失败', 'error'],
};

function bindImportControls() {
  const scanButton = $('#scanSourceButton');
  if (scanButton) scanButton.addEventListener('click', scanImportSource);
  const startButton = $('#importStartButton');
  if (startButton) startButton.addEventListener('click', startImportRun);
}

async function scanImportSource() {
  const source = $('#importSource').value.trim();
  const result = $('#scanResult');
  const hint = $('#importHint');
  if (!source) { hint.textContent = '请先填写源目录路径。'; return; }
  hint.textContent = '正在扫描目录…';
  result.hidden = true;
  try {
    const payload = await fetch('/api/platform/imports/scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source_dir: source }),
    });
    const data = await payload.json().catch(() => ({}));
    if (!payload.ok) throw new Error(data.detail || `HTTP ${payload.status}`);
    result.replaceChildren();
    const summary = document.createElement('p');
    summary.textContent = `图片 ${formatNumber(data.images.length)} · 视频 ${formatNumber(data.videos.length)} · SRT ${formatNumber(data.srt.length)}`;
    result.appendChild(summary);
    data.images.slice(0, 6).forEach((item) => {
      const row = document.createElement('div');
      row.textContent = `📷 ${item.name}`;
      result.appendChild(row);
    });
    data.videos.slice(0, 3).forEach((item) => {
      const row = document.createElement('div');
      row.textContent = `🎬 ${item.name}`;
      result.appendChild(row);
    });
    hint.textContent = '扫描完成，可以开始导入。';
    result.hidden = false;
  } catch (error) {
    hint.textContent = `扫描失败：${error.message}`;
  }
}

async function startImportRun() {
  const source = $('#importSource').value.trim();
  if (!source) { $('#importHint').textContent = '请先填写源目录路径。'; return; }
  const hint = $('#importHint');
  hint.textContent = '正在创建导入任务…';
  try {
    const payload = await fetch('/api/platform/imports', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source_dir: source,
        name: $('#importName').value.trim() || null,
        frame_stride: Number($('#frameStride').value || 30),
      }),
    });
    const data = await payload.json().catch(() => ({}));
    if (!payload.ok) throw new Error(data.detail || `HTTP ${payload.status}`);
    hint.textContent = `导入任务已启动：${data.run_id}`;
    loadImportRuns();
  } catch (error) {
    hint.textContent = `导入失败：${error.message}`;
  }
}

function renderImportRuns(runs) {
  const table = $('#importRunsTable');
  if (!table) return;
  table.replaceChildren();
  const head = document.createElement('div');
  head.className = 'console-row console-row-head';
  head.setAttribute('role', 'row');
  ['导入任务', '状态', '素材 / 地理参考', '停车分析', '操作'].forEach((label) => {
    const span = document.createElement('span');
    span.textContent = label;
    head.appendChild(span);
  });
  table.appendChild(head);
  if (!runs.length) {
    const empty = document.createElement('div');
    empty.className = 'console-row';
    const text = document.createElement('span');
    text.textContent = '暂无导入任务，填写上方源目录开始。';
    empty.appendChild(text);
    table.appendChild(empty);
    return;
  }
  runs.slice().reverse().forEach((run) => {
    const row = document.createElement('div');
    row.className = 'console-row';
    row.setAttribute('role', 'row');
    const [statusLabel, statusClass] = importStatusPresentation[run.status] || [run.status, 'pending'];
    const errorText = run.error ? `（${run.error}）` : '';
    const analyzed = run.parking_analyzed > 0;
    const nameCell = document.createElement('div');
    const name = document.createElement('b');
    name.textContent = run.name || run.run_id;
    nameCell.appendChild(name);
    const id = document.createElement('code');
    id.textContent = run.run_id;
    nameCell.appendChild(id);
    row.appendChild(nameCell);
    const statusCell = document.createElement('span');
    statusCell.className = `state ${statusClass}`;
    statusCell.textContent = statusLabel + errorText;
    row.appendChild(statusCell);
    const countCell = document.createElement('span');
    countCell.textContent = `${formatNumber(run.assets)} 素材 / ${formatNumber(run.georeferenced_assets)} 定位`;
    row.appendChild(countCell);
    const analysisCell = document.createElement('span');
    analysisCell.className = 'state ' + (analyzed ? 'ready' : 'pending');
    analysisCell.textContent = analyzed ? `已分析 ${formatNumber(run.parking_analyzed)}` : '未分析';
    row.appendChild(analysisCell);
    const actions = document.createElement('div');
    actions.className = 'model-actions';
    if (run.status === 'done' && run.assets > 0) {
      const analyze = document.createElement('button');
      analyze.className = 'button';
      analyze.type = 'button';
      analyze.textContent = '运行停车分析';
      analyze.addEventListener('click', () => startParkingAnalysis(run.run_id));
      actions.appendChild(analyze);
    }
    const mapLink = document.createElement('a');
    mapLink.className = 'button';
    mapLink.href = '/';
    mapLink.textContent = '地图查看';
    actions.appendChild(mapLink);
    row.appendChild(actions);
    table.appendChild(row);
  });
}

async function loadImportRuns() {
  try {
    const payload = await getJson('/api/platform/imports');
    renderImportRuns(payload.runs);
  } catch (error) {
    const table = $('#importRunsTable');
    if (table) table.textContent = `导入任务读取失败：${error.message}`;
  }
}

async function startParkingAnalysis(runId) {
  try {
    const response = await fetch(`/api/platform/imports/${encodeURIComponent(runId)}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ device: 'auto' }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    $('#importHint').textContent = `停车分析已启动：${runId}，完成后刷新可见。`;
  } catch (error) {
    $('#importHint').textContent = `停车分析启动失败：${error.message}`;
  }
}

async function loadMapPage() {
  const summary = await getJson('/api/platform/summary');
  $('#sourceLabel').textContent = `${summary.imagery.source} 已连接`;
  $('#imageryCount').textContent = formatNumber(summary.imagery.count);
  $('#countHigh').textContent = formatNumber(summary.parking_candidates.vehicle_row_supported);
  $('#countMedium').textContent = formatNumber(summary.parking_candidates.vehicle_detected);
  $('#countLow').textContent = formatNumber(summary.parking_candidates.segformer_only);
  const capabilities = $('#capabilityList');
  summary.capabilities.forEach((capability) => {
    const button = document.createElement('button');
    const name = document.createElement('span');
    const status = document.createElement('small');
    button.className = `capability ${capability.id === 'parking' ? 'active' : ''}`;
    button.disabled = capability.status === 'not_connected';
    name.textContent = capability.name;
    status.textContent = capability.status === 'not_connected' ? '待接入' : '查看图层';
    button.append(name, status);
    capabilities.appendChild(button);
  });

  const [west, south, east, north] = summary.overview.bounds;
  const imageBounds = [[south, west], [north, east]];
  const map = L.map('map', { crs: L.CRS.EPSG4326, zoomControl: false, attributionControl: false, minZoom: 8, maxZoom: 20 });
  L.control.zoom({ position: 'bottomleft' }).addTo(map);
  const overviewLayer = L.imageOverlay(summary.overview.image_url, imageBounds, { opacity: 1, alt: '全域正射影像概览' }).addTo(map);
  map.fitBounds(imageBounds, { padding: [18, 18] });
  map.createPane('nativeImagery');
  map.getPane('nativeImagery').style.zIndex = 350;
  map.getPane('nativeImagery').style.pointerEvents = 'none';
  const nativeImageryLayer = L.layerGroup().addTo(map);
  const candidateLayer = L.geoJSON([], { onEachFeature: bindCandidate, style: styleCandidate }).addTo(map);
  const importLayer = L.layerGroup().addTo(map);
  let activeImportRunId = null;
  let requestVersion = 0;
  let blockRequestVersion = 0;
  let candidateCount = 0;
  let nativeBlockCount = null;

  function importPointLayer(feature, latlng) {
    return L.circleMarker(latlng, { radius: 5.5, color: '#7c5cff', weight: 1.5, fillColor: '#7c5cff', fillOpacity: .75 });
  }

  function bindImportAsset(feature, layer) {
    const properties = feature.properties;
    const kindLabel = properties.kind === 'frame' ? '视频抽帧' : 'DJI 图片';
    layer.bindTooltip(kindLabel, { sticky: true });
    layer.bindPopup(buildImportPopup(activeImportRunId, properties), { maxWidth: 340 });
  }

  async function refreshImportAssets() {
    importLayer.clearLayers();
    const toggle = $('#importLayerToggle');
    if (!toggle || !toggle.checked) { $('#importAssetCount').textContent = '—'; return; }
    try {
      const payload = await getJson('/api/platform/imports');
      const ready = (payload.runs || []).filter((run) => run.status === 'done' && run.assets > 0);
      if (!ready.length) { $('#importAssetCount').textContent = '0'; return; }
      activeImportRunId = ready[ready.length - 1].run_id;
      const collection = await getJson(`/api/platform/imports/${encodeURIComponent(activeImportRunId)}/geojson`);
      L.geoJSON(collection, { pointToLayer: importPointLayer, onEachFeature: bindImportAsset }).addTo(importLayer);
      $('#importAssetCount').textContent = formatNumber(collection.features.length);
    } catch (error) {
      $('#importAssetCount').textContent = '—';
      console.error(error);
    }
  }

  function renderMapStatus() {
    if (nativeBlockCount === null) {
      $('#mapStatus').textContent = `${formatNumber(candidateCount)} 个候选 · 放大至 16 级查看原始影像`;
      return;
    }
    $('#mapStatus').textContent = `${formatNumber(nativeBlockCount)} 个原始图块 · ${formatNumber(candidateCount)} 个候选`;
  }

  function styleCandidate(feature) {
    const presentation = supportPresentation[feature.properties.support_level] || supportPresentation.segformer_only;
    return { color: presentation.color, weight: 1.4, fillColor: presentation.color, fillOpacity: .34 };
  }

  function bindCandidate(feature, layer) {
    layer.on('click', () => {
      map.fitBounds(layer.getBounds(), { padding: [80, 80], maxZoom: 19 });
      showCandidate(feature.properties.aoi_id);
    });
    layer.bindTooltip(supportPresentation[feature.properties.support_level]?.label || '停车候选', { sticky: true });
  }

  async function refreshCandidates() {
    const version = ++requestVersion;
    const levels = [...document.querySelectorAll('.evidence-filter input:checked')]
      .map((input) => input.value)
      .filter((value) => supportPresentation[value]);
    candidateLayer.clearLayers();
    if (!levels.length) { $('#mapStatus').textContent = '未选择候选图层'; return; }
    $('#mapStatus').textContent = '正在载入候选';
    const bounds = map.getBounds();
    const collections = await Promise.all(levels.map((support) => getJson(`/api/platform/candidates?west=${bounds.getWest()}&south=${bounds.getSouth()}&east=${bounds.getEast()}&north=${bounds.getNorth()}&support=${support}&limit=2000`)));
    if (version !== requestVersion) return;
    const features = collections.flatMap((collection) => collection.features);
    candidateLayer.addData({ type: 'FeatureCollection', features });
    candidateCount = features.length;
    renderMapStatus();
  }

  async function refreshNativeImagery() {
    const version = ++blockRequestVersion;
    if (map.getZoom() < 16) {
      nativeImageryLayer.clearLayers();
      overviewLayer.setOpacity(1);
      nativeBlockCount = null;
      renderMapStatus();
      return;
    }
    const bounds = map.getBounds();
    const query = `west=${bounds.getWest()}&south=${bounds.getSouth()}&east=${bounds.getEast()}&north=${bounds.getNorth()}&limit=256`;
    const payload = await getJson(`/api/platform/imagery/blocks?${query}`);
    if (version !== blockRequestVersion) return;
    nativeImageryLayer.clearLayers();
    payload.blocks.forEach((block) => {
      const [blockWest, blockSouth, blockEast, blockNorth] = block.bounds;
      L.imageOverlay(block.image_url, [[blockSouth, blockWest], [blockNorth, blockEast]], {
        pane: 'nativeImagery',
        opacity: 1,
        alt: block.block_id,
      }).addTo(nativeImageryLayer);
    });
    overviewLayer.setOpacity(.12);
    nativeBlockCount = payload.blocks.length;
    renderMapStatus();
  }

  async function refreshMap() {
    await Promise.all([refreshCandidates(), refreshNativeImagery()]);
  }

  let refreshTimer;
  map.on('moveend', () => { clearTimeout(refreshTimer); refreshTimer = setTimeout(refreshMap, 180); });
  document.querySelectorAll('.evidence-filter input').forEach((input) => input.addEventListener('change', refreshCandidates));
  const importToggle = $('#importLayerToggle');
  if (importToggle) importToggle.addEventListener('change', refreshImportAssets);
  await refreshMap();
  await refreshImportAssets();
}

async function showCandidate(aoiId) {
  const detail = await getJson(`/api/platform/candidates/${encodeURIComponent(aoiId)}`);
  const presentation = supportPresentation[detail.feature.properties.support_level];
  document.body.classList.add('candidate-selected');
  $('#emptyDetail').hidden = true;
  $('#candidateDetail').hidden = false;
  $('#detailTitle').textContent = aoiId;
  $('#detailBadge').textContent = presentation.label;
  $('#detailBadge').className = `status-badge ${presentation.className}`;
  const imageUrl = `/api/platform/media/image/${encodeURIComponent(aoiId)}`;
  const maskUrl = `/api/platform/media/mask/${encodeURIComponent(aoiId)}`;
  $('#detailImage').src = imageUrl;
  $('#detailImageBase').src = imageUrl;
  $('#detailMask').src = maskUrl;
  $('#downloadMask').href = maskUrl;
  const evidencePanel = $('#secondaryEvidence');
  const evidenceList = $('#secondaryEvidenceList');
  evidencePanel.hidden = true;
  evidenceList.replaceChildren();
  try {
    const secondary = await getJson(`/api/platform/candidates/${encodeURIComponent(aoiId)}/secondary`);
    const labels = { vehicle: '车辆', parking_marking: '停车标线', internal_aisle: '内部通道', building: '建筑', vegetation: '绿化' };
    Object.entries(secondary.evidence).forEach(([evidenceType, evidence]) => {
      const button = document.createElement('button');
      const count = document.createElement('span');
      button.type = 'button';
      const evidenceLabel = labels[evidenceType] || evidenceType;
      button.textContent = evidenceLabel;
      count.textContent = formatNumber(evidence.target_count);
      button.appendChild(count);
      button.addEventListener('click', () => openSecondaryEvidence(aoiId, evidenceType, evidenceLabel));
      evidenceList.appendChild(button);
    });
    evidencePanel.hidden = false;
  } catch (error) {
    if (!String(error.message).includes('404')) console.error(error);
  }
}

function buildImportPopup(runId, properties) {
  const box = document.createElement('div');
  box.className = 'import-popup';
  const image = document.createElement('img');
  image.src = `/api/platform/imports/${encodeURIComponent(runId)}/assets/${encodeURIComponent(properties.asset_id)}/image`;
  image.alt = '导入素材影像';
  box.appendChild(image);
  const meta = document.createElement('p');
  meta.textContent = `${properties.kind === 'frame' ? '视频抽帧' : 'DJI 图片'} · ${properties.gps_source}`;
  box.appendChild(meta);
  if (properties.captured_at) {
    const captured = document.createElement('p');
    captured.textContent = `时间 ${properties.captured_at}`;
    box.appendChild(captured);
  }
  if (properties.video) {
    const video = document.createElement('p');
    video.textContent = `视频 ${properties.video.name} @ ${properties.video.time_seconds}s`;
    box.appendChild(video);
  }
  if (properties.parking_fraction !== null && properties.parking_fraction !== undefined) {
    const parking = document.createElement('p');
    parking.textContent = `停车像素比例 ${(properties.parking_fraction * 100).toFixed(1)}%（模型候选）`;
    box.appendChild(parking);
  }
  const actions = document.createElement('div');
  actions.className = 'model-actions';
  const viewButton = document.createElement('button');
  viewButton.type = 'button';
  viewButton.className = 'button';
  viewButton.textContent = '查看影像';
  viewButton.addEventListener('click', () => openImportAssetViewer(runId, properties.asset_id, false));
  actions.appendChild(viewButton);
  if (properties.mask_available) {
    const maskButton = document.createElement('button');
    maskButton.type = 'button';
    maskButton.className = 'button';
    maskButton.textContent = '查看掩膜';
    maskButton.addEventListener('click', () => openImportAssetViewer(runId, properties.asset_id, true));
    actions.appendChild(maskButton);
  }
  box.appendChild(actions);
  return box;
}

function openImportAssetViewer(runId, assetId, showMask) {
  const base = `/api/platform/imports/${encodeURIComponent(runId)}/assets/${encodeURIComponent(assetId)}/image`;
  const mask = `/api/platform/imports/${encodeURIComponent(runId)}/assets/${encodeURIComponent(assetId)}/mask`;
  $('#viewerTitle').textContent = showMask ? 'SegFormer 停车区域 · 导入素材' : '导入素材影像';
  $('#viewerBase').src = base;
  $('#viewerOverlay').src = mask;
  $('#viewerOverlay').hidden = !showMask;
  resetMediaViewer();
  $('#mediaViewer').showModal();
}

function openSecondaryEvidence(aoiId, evidenceType, title) {
  $('#viewerTitle').textContent = `SAM3 · ${title}`;
  $('#viewerBase').src = $('#detailImage').src;
  $('#viewerOverlay').src = `/api/platform/candidates/${encodeURIComponent(aoiId)}/secondary/${encodeURIComponent(evidenceType)}`;
  $('#viewerOverlay').hidden = false;
  resetMediaViewer();
  $('#mediaViewer').showModal();
}

function closeCandidate() {
  document.body.classList.remove('candidate-selected');
  $('#candidateDetail').hidden = true;
  $('#emptyDetail').hidden = false;
}

const viewerState = { zoom: 1, x: 0, y: 0, dragging: false, pointerX: 0, pointerY: 0 };

function applyViewerTransform() {
  $('#viewerCanvas').style.transform = `translate3d(${viewerState.x}px, ${viewerState.y}px, 0) scale(${viewerState.zoom})`;
  $('#viewerReset').textContent = `${Math.round(viewerState.zoom * 100)}%`;
}

function resetMediaViewer() {
  Object.assign(viewerState, { zoom: 1, x: 0, y: 0, dragging: false });
  applyViewerTransform();
}

function setViewerZoom(nextZoom) {
  viewerState.zoom = Math.min(8, Math.max(1, nextZoom));
  if (viewerState.zoom === 1) {
    viewerState.x = 0;
    viewerState.y = 0;
  }
  applyViewerTransform();
}

function openMediaViewer(mode) {
  const showMask = mode === 'mask';
  $('#viewerTitle').textContent = showMask ? 'SegFormer 区域' : '原始影像';
  $('#viewerBase').src = showMask ? $('#detailImageBase').src : $('#detailImage').src;
  $('#viewerOverlay').src = $('#detailMask').src;
  $('#viewerOverlay').hidden = !showMask;
  resetMediaViewer();
  $('#mediaViewer').showModal();
  $('#viewerViewport').focus();
}

function configureMediaViewer() {
  const dialog = $('#mediaViewer');
  const viewport = $('#viewerViewport');
  document.querySelectorAll('[data-viewer]').forEach((button) => button.addEventListener('click', () => openMediaViewer(button.dataset.viewer)));
  $('#viewerClose').addEventListener('click', () => dialog.close());
  $('#viewerZoomIn').addEventListener('click', () => setViewerZoom(viewerState.zoom * 1.4));
  $('#viewerZoomOut').addEventListener('click', () => setViewerZoom(viewerState.zoom / 1.4));
  $('#viewerReset').addEventListener('click', resetMediaViewer);
  viewport.addEventListener('wheel', (event) => {
    event.preventDefault();
    setViewerZoom(viewerState.zoom * (event.deltaY < 0 ? 1.2 : 1 / 1.2));
  }, { passive: false });
  viewport.addEventListener('dblclick', () => setViewerZoom(viewerState.zoom > 1 ? 1 : 3));
  viewport.addEventListener('pointerdown', (event) => {
    viewerState.dragging = true;
    viewerState.pointerX = event.clientX;
    viewerState.pointerY = event.clientY;
    viewport.setPointerCapture(event.pointerId);
  });
  viewport.addEventListener('pointermove', (event) => {
    if (!viewerState.dragging || viewerState.zoom === 1) return;
    viewerState.x += event.clientX - viewerState.pointerX;
    viewerState.y += event.clientY - viewerState.pointerY;
    viewerState.pointerX = event.clientX;
    viewerState.pointerY = event.clientY;
    applyViewerTransform();
  });
  viewport.addEventListener('pointerup', () => { viewerState.dragging = false; });
  viewport.addEventListener('pointercancel', () => { viewerState.dragging = false; });
}

document.addEventListener('DOMContentLoaded', () => {
  const page = document.body.dataset.page;
  const task = page === 'map' ? loadMapPage() : loadEngineering();
  task.catch((error) => {
    console.error(error);
    document.body.dataset.loadError = 'true';
    const mapStatus = $('#mapStatus');
    if (mapStatus) mapStatus.textContent = '数据载入失败，请联系技术人员';
  });
  const dialog = $('#analysisDialog');
  if (dialog) $('#analysisButton').addEventListener('click', () => dialog.showModal());
  const closeDetail = $('#closeDetail');
  if (closeDetail) closeDetail.addEventListener('click', closeCandidate);
  if ($('#mediaViewer')) configureMediaViewer();
});
