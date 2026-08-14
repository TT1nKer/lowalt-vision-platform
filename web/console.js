(function () {
  'use strict';

  let overview = null;
  let pendingTaskConfig = null;

  function number(value) {
    return Number(value || 0).toLocaleString('zh-CN');
  }

  function setDot(selector, healthy, enabled = true) {
    const dot = $(selector);
    if (!dot) return;
    dot.className = 'modern-status-dot ' + (healthy ? 'ok' : enabled ? 'warn' : '');
  }

  function parseParams(button) {
    try {
      return JSON.parse(button.dataset.jobParams || '{}');
    } catch (_) {
      toast('操作参数无效，请联系维护人员', 'err');
      return null;
    }
  }

  function chooseNextAction(data) {
    const total = Number(data.total_index || data.target_count || 0);
    const reviewed = Number(data.reviewed_count || 0);
    const labels = data.label_counts || {};
    const positive = Number(labels.accept || 0) + Number(labels.hard_positive || 0);
    if (!data.index_exists || total === 0) {
      return { label: '运行目标识别', action: () => startJob('p1_all') };
    }
    if (reviewed < total) {
      return { label: '筛选候选结果', action: () => location.assign('/review') };
    }
    if (!data.yolo_ready && positive > 0) {
      return { label: '生成训练数据', action: () => startJob('p3_export', { fmt: 'seg' }) };
    }
    return { label: '验证或训练模型', action: openTrainingSettings };
  }

  function renderOverview(data) {
    overview = data;
    const labels = data.label_counts || {};
    const total = Number(data.total_index || data.target_count || 0);
    const reviewed = Number(data.reviewed_count || 0);
    const positive = Number(labels.accept || 0) + Number(labels.hard_positive || 0);
    const source = String(data.image_dir || '').split(/[\\/]/).filter(Boolean).pop() || '未配置';
    const concept = String(data.text_prompt || '未配置目标');
    const autoCount = Number(data.gemma_reviewed || 0);
    const humanCount = Number(data.human_reviewed || 0);
    const percent = total ? Math.round(reviewed * 100 / total) : 0;

    $('#projectName').textContent = source;
    $('#activeTaskTitle').textContent = concept;
    $('#pageSummary').textContent = `数据源：${source} · 识别目标：${concept}`;
    $('#runLabel').textContent = `任务空间：${concept}`;
    $('#targetCount').textContent = number(total);
    $('#reviewedCount').textContent = number(reviewed);
    $('#positiveCount').textContent = number(positive);
    $('#reviewNote').textContent = `自动 ${number(autoCount)} · 人工 ${number(humanCount)} · ${percent}%`;
    $('#datasetState').textContent = data.yolo_ready ? '已生成' : '待生成';
    $('#datasetNote').textContent = data.yolo_ready ? '可进入质量检查与训练' : '确认结果后生成';
    $('#indexState').textContent = data.index_exists ? `${number(total)} 个目标` : '尚未建立';
    $('#exportState').textContent = data.yolo_ready ? '已生成' : '等待生成';
    setDot('#indexDot', Boolean(data.index_exists));
    setDot('#exportDot', Boolean(data.yolo_ready));

    const next = chooseNextAction(data);
    const button = $('#nextActionButton');
    $('#nextActionText').textContent = next.label;
    button.textContent = next.label;
    button.disabled = false;
    button.onclick = next.action;

    if (data.train_defaults) {
      $('#train_model').value = data.train_defaults.model || $('#train_model').value;
      $('#train_imgsz').value = data.train_defaults.imgsz || $('#train_imgsz').value;
      $('#train_epochs').value = data.train_defaults.epochs || $('#train_epochs').value;
    }
  }

  async function loadOverview() {
    try {
      renderOverview(await apiGet('/api/overview'));
    } catch (error) {
      $('#pageSummary').textContent = '无法读取当前任务，请检查平台服务。';
      $('#nextActionText').textContent = '服务状态异常';
    }
  }

  async function loadHealth() {
    try {
      const health = await apiGet('/api/health');
      setDot('#samDot', Boolean(health.sam));
      setDot('#gemmaDot', Boolean(health.gemma), Boolean(health.gemma_enabled));
      $('#samState').textContent = health.sam ? '已连接' : '不可用';
      $('#gemmaState').textContent = !health.gemma_enabled ? '未启用' : health.gemma ? '已连接' : '不可用';
      $('#healthText').textContent = health.sam ? '平台运行正常' : '部分服务不可用';
    } catch (_) {
      $('#healthText').textContent = '平台服务异常';
      $('#serviceState').textContent = '连接异常';
    }
  }

  async function startGemma() {
    const result = await apiPost('/api/gemma/auto/start', {});
    toast(result.ok === false ? result.error || '启动失败' : 'Gemma 筛选已启动', result.ok === false ? 'err' : 'ok');
  }

  async function stopGemma() {
    const result = await apiPost('/api/gemma/auto/stop', {});
    toast(result.ok === false ? result.error || '停止失败' : '已请求停止 Gemma', result.ok === false ? 'err' : 'info');
  }

  function openTaskDialog() {
    const dialog = $('#taskDialog');
    if (typeof dialog.showModal === 'function') dialog.showModal();
    else dialog.setAttribute('open', '');
  }

  function summarizeConfig(config) {
    const sam = config.sam3 || {};
    const gemma = config.gemma || {};
    const prompts = Array.isArray(sam.prompts) ? sam.prompts : [sam.text_prompt].filter(Boolean);
    const types = Array.isArray(gemma.object_types) ? gemma.object_types : [];
    const preview = $('#taskConfigPreview');
    clear(preview);
    preview.appendChild(el('div', { class: 'vision-preview-row' }, [el('span', { text: '识别提示词' }), el('strong', { text: prompts.join('、') || '未生成' })]));
    preview.appendChild(el('div', { class: 'vision-preview-row' }, [el('span', { text: '目标类别' }), el('strong', { text: types.join('、') || '按目标描述判断' })]));
    preview.hidden = false;
  }

  async function generateTaskConfig() {
    const domain = $('#taskDomain').value.trim();
    if (!domain) {
      toast('请先描述需要识别的对象', 'info');
      $('#taskDomain').focus();
      return;
    }
    const button = $('#generateTaskButton');
    button.disabled = true;
    button.textContent = '正在生成…';
    try {
      const result = await apiPost('/api/gemma/generate-config', { domain });
      if (result.ok === false) throw new Error(result.error || '生成失败');
      const generated = result.config || result;
      const sam3 = Object.assign({}, generated.sam3 || {});
      const prompts = Array.isArray(sam3.prompts) ? sam3.prompts.filter(Boolean) : [];
      if (!sam3.text_prompt && prompts.length) sam3.text_prompt = prompts[0];
      pendingTaskConfig = {
        sam3,
        gemma: generated.gemma || {},
        yolo_obb: generated.yolo_obb || {},
      };
      summarizeConfig(pendingTaskConfig);
      $('#applyTaskButton').disabled = false;
    } catch (error) {
      toast('配置生成失败：' + error.message, 'err');
    } finally {
      button.disabled = false;
      button.textContent = '生成任务配置';
    }
  }

  async function applyTaskConfig() {
    if (!pendingTaskConfig) return;
    if (!confirm('应用后会切换当前识别目标。已有结果不会自动删除，也不会立即运行识别。确认继续？')) return;
    const result = await apiPost('/api/gemma/apply-config', { config: pendingTaskConfig });
    if (result.ok === false) {
      toast(result.error || '应用失败', 'err');
      return;
    }
    const selectedPrompt = pendingTaskConfig.sam3 && pendingTaskConfig.sam3.text_prompt;
    if (selectedPrompt) {
      const selected = await apiPost('/api/runs/select', { prompt: selectedPrompt });
      if (selected.ok === false) {
        toast('配置已保存，但任务视图切换失败：' + (selected.error || '未知错误'), 'err');
        return;
      }
    }
    toast('识别目标已更新', 'ok');
    $('#taskDialog').close();
    pendingTaskConfig = null;
    await loadOverview();
  }

  function openTrainingSettings() {
    const details = $('#trainingSettings details');
    details.open = true;
    details.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'center' });
  }

  function startTraining(resume) {
    if (resume) {
      startJob('p3_train', { resume: true });
      return;
    }
    const imgsz = Number($('#train_imgsz').value);
    const epochs = Number($('#train_epochs').value);
    if (!Number.isInteger(imgsz) || imgsz < 128 || !Number.isInteger(epochs) || epochs < 1) {
      toast('请输入有效的输入尺寸和训练轮数', 'err');
      return;
    }
    startJob('p3_train', { model: $('#train_model').value.trim(), imgsz, epochs });
  }

  function bindControls() {
    $all('[data-job]').forEach((button) => button.addEventListener('click', () => {
      const params = parseParams(button);
      if (params !== null) startJob(button.dataset.job, params);
    }));
    $all('[data-command="configure"]').forEach((button) => button.addEventListener('click', openTaskDialog));
    $all('[data-command="gemma-start"]').forEach((button) => button.addEventListener('click', startGemma));
    $all('[data-command="gemma-stop"]').forEach((button) => button.addEventListener('click', stopGemma));
    $all('[data-command="train-open"]').forEach((button) => button.addEventListener('click', openTrainingSettings));
    $('#configureTaskButton').addEventListener('click', openTaskDialog);
    $('#generateTaskButton').addEventListener('click', generateTaskConfig);
    $('#applyTaskButton').addEventListener('click', applyTaskConfig);
    $('#cancelJobButton').addEventListener('click', cancelActiveJob);
    $('#startTrainButton').addEventListener('click', () => startTraining(false));
    $('#resumeTrainButton').addEventListener('click', () => startTraining(true));
  }

  window.onJobDone = function () {
    loadOverview();
  };

  const originalSetBar = window._setBar;
  window._setBar = function (progress, status) {
    originalSetBar(progress, status);
    const labels = { pending: '等待中', running: '运行中', done: '已完成', error: '执行失败', cancelled: '已取消' };
    $('#jobState').textContent = labels[status] || '待命';
  };

  bindControls();
  loadOverview();
  loadHealth();
  resumeLatestJob();
  setInterval(loadOverview, 5000);
  setInterval(loadHealth, 15000);
})();
