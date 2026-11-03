/* ============================================================
   OMEGA Utils — cliente da interface
   Gráficos leves em <canvas> (sem dependências externas) e
   roteamento dos formulários para a API Flask.
   ============================================================ */
"use strict";

/* ---------------- utilidades ---------------- */

const $ = (sel, raiz = document) => raiz.querySelector(sel);
const $$ = (sel, raiz = document) => [...raiz.querySelectorAll(sel)];

async function postJson(url, dados) {
    const resposta = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(dados),
    });
    let corpo;
    try {
        corpo = await resposta.json();
    } catch {
        throw new Error(`resposta inválida do servidor (HTTP ${resposta.status})`);
    }
    if (!corpo.ok) throw new Error(corpo.erro || `HTTP ${resposta.status}`);
    return corpo;
}

async function getJson(url) {
    const resposta = await fetch(url);
    const corpo = await resposta.json();
    if (!corpo.ok) throw new Error(corpo.erro || `HTTP ${resposta.status}`);
    return corpo;
}

function escutaHtml(texto) {
    return String(texto ?? "")
        .replace(/&/g, "&amp;").replace(/</g, "&lt;")
        .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

/* ---------------- biblioteca de gráficos ---------------- */

const PALETA = ["#7c9aff", "#b07cff", "#3fd6c6", "#ffc46b", "#ff7ca8", "#5ee6a8"];

function _limitesSuaves(min, max) {
    if (!isFinite(min) || !isFinite(max)) return [0, 1];
    if (min === max) { min -= 1; max += 1; }
    const margem = (max - min) * 0.08;
    min -= margem; max += margem;
    let passo = Math.pow(10, Math.floor(Math.log10((max - min) / 4)));
    const razao = (max - min) / passo;
    if (razao > 8) passo *= 2;
    if (razao < 2) passo /= 2;
    return [Math.floor(min / passo) * passo, Math.ceil(max / passo) * passo];
}

function _rotuloNumero(v) {
    if (v === 0) return "0";
    const a = Math.abs(v);
    if (a >= 1e5 || a < 1e-3) return v.toExponential(1);
    return parseFloat(v.toPrecision(4)).toString();
}

/**
 * Desenha uma ou mais séries {x:[], y:[], rotulo, cor} num canvas.
 * opts: {marcaX, marcaY, areaEntre, titulo}
 */
function desenharCurvas(canvas, series, opts = {}) {
    const caixa = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const largura = caixa.clientWidth - 24;
    const altura = opts.altura || 260;
    canvas.width = largura * dpr;
    canvas.height = altura * dpr;
    canvas.style.height = altura + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, largura, altura);

    const margem = { esq: 58, dir: 16, topo: 14, base: 34 };
    const w = largura - margem.esq - margem.dir;
    const h = altura - margem.topo - margem.base;

    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const s of series) {
        for (let i = 0; i < s.x.length; i++) {
            const x = s.x[i], y = s.y[i];
            if (y == null || !isFinite(y)) continue;
            if (x < xMin) xMin = x;
            if (x > xMax) xMax = x;
            if (y < yMin) yMin = y;
            if (y > yMax) yMax = y;
        }
    }
    if (!isFinite(xMin)) { ctx.restore?.(); return; }
    if (opts.forcarZeroY) yMin = Math.min(yMin, 0);
    [yMin, yMax] = _limitesSuaves(yMin, yMax);
    [xMin, xMax] = _limitesSuaves(xMin, xMax);

    const px = (x) => margem.esq + ((x - xMin) / (xMax - xMin)) * w;
    const py = (y) => margem.topo + h - ((y - yMin) / (yMax - yMin)) * h;

    /* grade */
    ctx.strokeStyle = "rgba(122,142,198,0.13)";
    ctx.fillStyle = "rgba(154,167,199,0.75)";
    ctx.font = "10.5px 'JetBrains Mono', monospace";
    ctx.lineWidth = 1;
    const linhas = 5;
    for (let i = 0; i <= linhas; i++) {
        const v = yMin + (i / linhas) * (yMax - yMin);
        const yy = py(v);
        ctx.beginPath(); ctx.moveTo(margem.esq, yy); ctx.lineTo(margem.esq + w, yy); ctx.stroke();
        ctx.textAlign = "right"; ctx.textBaseline = "middle";
        ctx.fillText(_rotuloNumero(v), margem.esq - 8, yy);
    }
    const colunas = 6;
    for (let i = 0; i <= colunas; i++) {
        const v = xMin + (i / colunas) * (xMax - xMin);
        const xx = px(v);
        ctx.beginPath(); ctx.moveTo(xx, margem.topo); ctx.lineTo(xx, margem.topo + h); ctx.stroke();
        ctx.textAlign = "center"; ctx.textBaseline = "top";
        ctx.fillText(_rotuloNumero(v), xx, margem.topo + h + 8);
    }

    /* eixo y = 0 destacado */
    if (yMin < 0 && yMax > 0) {
        ctx.strokeStyle = "rgba(122,142,198,0.4)";
        ctx.beginPath(); ctx.moveTo(margem.esq, py(0)); ctx.lineTo(margem.esq + w, py(0)); ctx.stroke();
    }

    /* área sombreada entre a e b (integrais) */
    if (opts.areaEntre) {
        const [a, b] = opts.areaEntre[0] < opts.areaEntre[1] ? opts.areaEntre : [opts.areaEntre[1], opts.areaEntre[0]];
        const s = series[0];
        ctx.save();
        ctx.beginPath();
        ctx.rect(px(a), margem.topo, px(b) - px(a), h);
        ctx.clip();
        ctx.beginPath();
        let comecou = false;
        for (let i = 0; i < s.x.length; i++) {
            if (s.y[i] == null || !isFinite(s.y[i])) continue;
            const X = px(s.x[i]), Y = py(s.y[i]);
            if (!comecou) { ctx.moveTo(X, Y); comecou = true; } else ctx.lineTo(X, Y);
        }
        ctx.lineTo(px(b), py(0)); ctx.lineTo(px(a), py(0)); ctx.closePath();
        ctx.fillStyle = "rgba(124,154,255,0.16)";
        ctx.fill();
        ctx.restore();
    }

    /* séries */
    series.forEach((s, indice) => {
        const cor = s.cor || PALETA[indice % PALETA.length];
        ctx.strokeStyle = cor;
        ctx.lineWidth = 2.1;
        ctx.lineJoin = "round";
        ctx.beginPath();
        let comecou = false;
        for (let i = 0; i < s.x.length; i++) {
            const y = s.y[i];
            if (y == null || !isFinite(y)) { comecou = false; continue; }
            const X = px(s.x[i]), Y = py(Math.max(yMin, Math.min(yMax, y)));
            if (!comecou) { ctx.moveTo(X, Y); comecou = true; } else ctx.lineTo(X, Y);
        }
        ctx.stroke();
    });

    /* marca (raiz, ponto de tangência…) */
    if (opts.marcaX != null && isFinite(opts.marcaX)) {
        const mx = px(opts.marcaX);
        ctx.strokeStyle = "rgba(255,124,168,0.75)";
        ctx.setLineDash([5, 4]);
        ctx.beginPath(); ctx.moveTo(mx, margem.topo); ctx.lineTo(mx, margem.topo + h); ctx.stroke();
        ctx.setLineDash([]);
        if (opts.marcaY != null && isFinite(opts.marcaY)) {
            ctx.fillStyle = "#ff7ca8";
            ctx.beginPath(); ctx.arc(mx, py(opts.marcaY), 4.5, 0, 2 * Math.PI); ctx.fill();
        }
    }

    /* legenda */
    if (series.some((s) => s.rotulo)) {
        let desloc = margem.esq + 8;
        ctx.font = "11px Inter, sans-serif";
        ctx.textBaseline = "middle"; ctx.textAlign = "left";
        series.forEach((s, indice) => {
            if (!s.rotulo) return;
            const cor = s.cor || PALETA[indice % PALETA.length];
            ctx.fillStyle = cor;
            ctx.fillRect(desloc, margem.topo + 8, 10, 3);
            ctx.fillStyle = "rgba(232,236,248,0.85)";
            ctx.fillText(s.rotulo, desloc + 15, margem.topo + 10);
            desloc += 22 + ctx.measureText(s.rotulo).width;
        });
    }
}

function desenharDispersao(canvas, pontos, opts = {}) {
    /* pontos: [{x, y, v}] — mapa de calor simples x·y com cor por v */
    const caixa = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const largura = caixa.clientWidth - 24;
    const altura = opts.altura || 300;
    canvas.width = largura * dpr;
    canvas.height = altura * dpr;
    canvas.style.height = altura + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, largura, altura);

    const margem = 26;
    const meioY = altura / 2;
    const escalaX = (largura - 2 * margem) / (opts.comprimento || 1);
    const escalaY = escalaX;

    /* contorno do bocal */
    if (opts.contorno) {
        ctx.fillStyle = "rgba(124,154,255,0.10)";
        ctx.strokeStyle = "rgba(124,154,255,0.6)";
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        opts.contorno.x.forEach((x, i) => {
            const X = margem + x * escalaX, Y = meioY - opts.contorno.superior[i] * escalaY;
            i === 0 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y);
        });
        for (let i = opts.contorno.x.length - 1; i >= 0; i--) {
            const X = margem + opts.contorno.x[i] * escalaX;
            ctx.lineTo(X, meioY + opts.contorno.superior[i] * escalaY);
        }
        ctx.closePath(); ctx.fill(); ctx.stroke();
    }

    const valores = pontos.map((p) => p.v).filter((v) => isFinite(v));
    const vMin = Math.min(...valores), vMax = Math.max(...valores);

    for (const p of pontos) {
        if (!isFinite(p.v)) continue;
        const t = vMax > vMin ? (p.v - vMin) / (vMax - vMin) : 0.5;
        const X = margem + p.x * escalaX;
        const Y = meioY - p.y * escalaY;
        ctx.fillStyle = _corCalor(t);
        ctx.globalAlpha = 0.85;
        ctx.beginPath(); ctx.arc(X, Y, opts.raio || 2.6, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.globalAlpha = 1;

    /* barra de escala */
    if (opts.rotuloEscala) {
        const bw = 150, bh = 8, bx = largura - bw - margem, by = altura - 18;
        const grad = ctx.createLinearGradient(bx, 0, bx + bw, 0);
        for (let i = 0; i <= 10; i++) grad.addColorStop(i / 10, _corCalor(i / 10));
        ctx.fillStyle = grad;
        ctx.fillRect(bx, by, bw, bh);
        ctx.strokeStyle = "rgba(122,142,198,0.4)";
        ctx.strokeRect(bx, by, bw, bh);
        ctx.fillStyle = "rgba(154,167,199,0.9)";
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.textAlign = "left"; ctx.textBaseline = "bottom";
        ctx.fillText(_rotuloNumero(vMin), bx, by - 3);
        ctx.textAlign = "right";
        ctx.fillText(`${_rotuloNumero(vMax)}  ·  ${opts.rotuloEscala}`, bx + bw, by - 3);
    }
}

function _corCalor(t) {
    /* azul → ciano → verde → âmbar → vermelho */
    const paradas = [
        [0.0, [56, 92, 231]], [0.25, [63, 214, 198]], [0.5, [94, 230, 168]],
        [0.75, [255, 196, 107]], [1.0, [255, 107, 129]],
    ];
    for (let i = 1; i < paradas.length; i++) {
        if (t <= paradas[i][0]) {
            const [t0, c0] = paradas[i - 1], [t1, c1] = paradas[i];
            const f = (t - t0) / (t1 - t0);
            const mix = c0.map((v, k) => Math.round(v + (c1[k] - v) * f));
            return `rgb(${mix[0]},${mix[1]},${mix[2]})`;
        }
    }
    return "rgb(255,107,129)";
}

/* ---------------- estrutura dos cartões ---------------- */

function criarResultado(raizCartao) {
    let caixa = $(".resultado", raizCartao);
    if (!caixa) {
        caixa = document.createElement("div");
        caixa.className = "resultado oculto";
        raizCartao.appendChild(caixa);
    }
    caixa.innerHTML = "";
    caixa.classList.remove("oculto");
    return caixa;
}

function mostrarErro(raiz, mensagem) {
    const caixa = criarResultado(raiz);
    caixa.innerHTML = `<div class="erro-caixa">⚠️ <span>${escutaHtml(mensagem)}</span></div>`;
}

function kpis(caixa, itens) {
    const grade = document.createElement("div");
    grade.className = "kpis";
    for (const item of itens) {
        if (item.valor == null) continue;
        const div = document.createElement("div");
        div.className = "kpi";
        div.innerHTML = `<div class="rotulo">${escutaHtml(item.rotulo)}</div>
                         <div class="valor ${String(item.valor).length > 16 ? "pequeno" : ""}">${escutaHtml(item.valor)}</div>`;
        grade.appendChild(div);
    }
    caixa.appendChild(grade);
    return grade;
}

function nota(caixa, texto) {
    if (!texto) return;
    const div = document.createElement("div");
    div.className = "nota";
    div.textContent = texto;
    caixa.appendChild(div);
}

function grafico(caixa, series, opts = {}) {
    if (!series.length || !series[0].x.length) return;
    const envolucro = document.createElement("div");
    envolucro.className = "grafico-caixa";
    const tela = document.createElement("canvas");
    envolucro.appendChild(tela);
    caixa.appendChild(envolucro);
    desenharCurvas(tela, series, opts);
    let redimensionar;
    window.addEventListener("resize", () => {
        clearTimeout(redimensionar);
        redimensionar = setTimeout(() => desenharCurvas(tela, series, opts), 150);
    });
    return tela;
}

async function executar(botao, funcaoAssincrona) {
    const rotuloOriginal = botao.innerHTML;
    botao.disabled = true;
    botao.innerHTML = `<span class="girando"></span>calculando…`;
    try {
        await funcaoAssincrona();
    } catch (erro) {
        mostrarErro(botao.closest(".ferramenta"), erro.message || String(erro));
    } finally {
        botao.disabled = false;
        botao.innerHTML = rotuloOriginal;
    }
}

/* ---------------- registros de ferramentas ---------------- */

const FERRAMENTAS = {};

function registrar(id, initFn) { FERRAMENTAS[id] = initFn; }

/* ---------- Integrais ---------- */

registrar("integrais", (cartao) => {
    $("form[data-api]", cartao).forEach((form) => {
        form.addEventListener("submit", (evento) => {
            evento.preventDefault();
            const botao = $('button[type="submit"]', form);
            executar(botao, async () => {
                const dados = Object.fromEntries(new FormData(form).entries());
                const resposta = await postJson(form.dataset.api, dados);
                const caixa = criarResultado(cartao);
                kpis(caixa, [
                    { rotulo: "Valor da integral", valor: resposta.valor },
                    ...(resposta.referencia_simpson ? [{ rotulo: "Ref. Simpson", valor: resposta.referencia_simpson }] : []),
                    ...(resposta.resumo && resposta.resumo.erro_padrao
                        ? [{ rotulo: "Erro padrão", valor: formatarCampo(resposta.resumo.erro_padrao) }] : []),
                ]);
                nota(caixa, resposta.detalhe);
                if (resposta.grafico) {
                    grafico(caixa, [{ x: resposta.grafico.x, y: resposta.grafico.y, rotulo: "f(x)", cor: "#7c9aff" }],
                        { areaEntre: resposta.grafico.a != null ? [resposta.grafico.a, resposta.grafico.b] : null,
                          marcaX: resposta.grafico.marca ? resposta.grafico.marca[0] : null,
                          forcarZeroY: true });
                }
            });
        });
    });
});

function formatarCampo(v) {
    if (typeof v === "number") return v.toPrecision(6);
    return String(v);
}

/* ---------- Derivadas ---------- */

registrar("derivadas", (cartao) => {
    const form = $("form", cartao);
    form.addEventListener("submit", (evento) => {
        evento.preventDefault();
        executar($("button", form), async () => {
            const dados = Object.fromEntries(new FormData(form).entries());
            const r = await postJson("/api/derivada", dados);
            const caixa = criarResultado(cartao);
            kpis(caixa, [
                { rotulo: "f′(x₀) centralizada", valor: r.centralizada },
                { rotulo: "f′(x₀) progressiva", valor: r.progressiva },
                { rotulo: "f″(x₀)", valor: r.segunda },
                { rotulo: "Reta tangente", valor: `m=${r.tangente.declive}, b=${r.tangente.intercepto}` },
            ]);
            nota(caixa, `f(x) = ${r.expressao} no ponto x₀ informado`);
            grafico(caixa, [
                { x: r.grafico.x, y: r.grafico.y, rotulo: "f(x)", cor: "#7c9aff" },
                { x: r.grafico.x, y: r.grafico.tangente, rotulo: "tangente", cor: "#ff7ca8" },
            ], { marcaX: r.grafico.ponto[0], marcaY: r.grafico.ponto[1] });
        });
    });
});

/* ---------- Equações & Física ---------- */

registrar("equacoes", (cartao) => {
    $$("form[data-api]", cartao).forEach((form) => {
        form.addEventListener("submit", (evento) => {
            evento.preventDefault();
            executar($('button[type="submit"]', form), async () => {
                const dados = Object.fromEntries(new FormData(form).entries());
                const r = await postJson(form.dataset.api, dados);
                const caixa = criarResultado(cartao);
                if (form.dataset.api.includes("grau2")) {
                    kpis(caixa, [
                        { rotulo: "Discriminante Δ", valor: r.delta },
                        { rotulo: "Raízes", valor: Array.isArray(r.raizes) ? r.raizes.join("  |  ") : String(r.raizes) },
                    ]);
                    nota(caixa, String(r.bruto ?? ""));
                    if (r.grafico) grafico(caixa, [{ x: r.grafico.x, y: r.grafico.y, rotulo: "ax²+bx+c" }], { forcarZeroY: true });
                } else if (form.dataset.api.includes("mrv")) {
                    kpis(caixa, [
                        { rotulo: "Posição final S(t)", valor: r.posicao + " m" },
                        { rotulo: "Velocidade final", valor: r.velocidade_final + " m/s" },
                    ]);
                    nota(caixa, r.detalhe);
                    if (r.grafico) {
                        grafico(caixa, [
                            { x: r.grafico.x, y: r.grafico.serie1, rotulo: r.grafico.rotulo1, cor: "#7c9aff" },
                            { x: r.grafico.x, y: r.grafico.serie2, rotulo: r.grafico.rotulo2, cor: "#3fd6c6" },
                        ]);
                    }
                } else if (form.dataset.api.includes("rk4")) {
                    kpis(caixa, [{ rotulo: "y no fim da integração", valor: r.final }]);
                    nota(caixa, r.detalhe);
                    if (r.grafico) grafico(caixa, [{ x: r.grafico.x, y: r.grafico.y, rotulo: "y(t)" }]);
                }
            });
        });
    });
});

/* ---------- Funções & Raízes ---------- */

registrar("funcoes", (cartao) => {
    $$("form[data-api]", cartao).forEach((form) => {
        form.addEventListener("submit", (evento) => {
            evento.preventDefault();
            executar($('button[type="submit"]', form), async () => {
                const dados = Object.fromEntries(new FormData(form).entries());
                const r = await postJson(form.dataset.api, dados);
                const caixa = criarResultado(cartao);
                if (form.dataset.api.includes("polimonio")) {
                    kpis(caixa, [{ rotulo: "Raízes reais", valor: r.raizes.join("  |  ") }]);
                } else if (form.dataset.api.includes("limite")) {
                    kpis(caixa, [{ rotulo: "Limite estimado", valor: r.limite }]);
                } else {
                    kpis(caixa, [
                        { rotulo: "Raiz encontrada", valor: r.raiz },
                        { rotulo: "|f(x)| na raiz", valor: r.f_na_raiz },
                    ]);
                }
                nota(caixa, r.detalhe);
                if (r.grafico) {
                    grafico(caixa, [{ x: r.grafico.x, y: r.grafico.y, rotulo: "f(x)" }],
                        { marcaX: r.grafico.marca ? r.grafico.marca[0] : null,
                          marcaY: r.grafico.marca ? r.grafico.marca[1] : null,
                          forcarZeroY: true });
                }
            });
        });
    });
});

/* ---------- Teoremas vetoriais ---------- */

registrar("teoremas", (cartao) => {
    const form = $("form", cartao);
    form.addEventListener("submit", (evento) => {
        evento.preventDefault();
        executar($("button", form), async () => {
            const dados = Object.fromEntries(new FormData(form).entries());
            const r = await postJson("/api/teorema/green", dados);
            const caixa = criarResultado(cartao);
            kpis(caixa, [
                { rotulo: "∮ P dx + Q dy", valor: r.resultado.integral_linha },
                { rotulo: "∬ rotacional · dA", valor: r.resultado.integral_dupla },
                { rotulo: "Diferença", valor: r.resultado.diferenca_absoluta },
            ]);
            nota(caixa, r.detalhe + " — os dois lados devem coincidir dentro da precisão numérica.");
        });
    });
});

/* ---------- Monte Carlo ---------- */

registrar("montecarlo", (cartao) => {
    const form = $("form", cartao);
    form.addEventListener("submit", (evento) => {
        evento.preventDefault();
        executar($("button", form), async () => {
            const dados = Object.fromEntries(new FormData(form).entries());
            const r = await postJson("/api/montecarlo/integracao", dados);
            const caixa = criarResultado(cartao);
            kpis(caixa, [
                { rotulo: "Integral (Monte Carlo)", valor: r.valor },
                { rotulo: "Erro padrão", valor: r.resumo.erro_padrao ?? null },
                { rotulo: "IC 95% inferior", valor: r.resumo.intervalo_confianca_95 ? r.resumo.intervalo_confianca_95[0] : null },
                { rotulo: "IC 95% superior", valor: r.resumo.intervalo_confianca_95 ? r.resumo.intervalo_confianca_95[1] : null },
                { rotulo: "Ref. Simpson", valor: r.referencia_simpson },
            ]);
            nota(caixa, r.detalhe);
            if (r.grafico) {
                grafico(caixa, [{ x: r.grafico.x, y: r.grafico.y, rotulo: "f(x)" }],
                    { areaEntre: [r.grafico.a, r.grafico.b], forcarZeroY: true });
            }
        });
    });
});

/* ---------- Estruturas ---------- */

registrar("estruturas", (cartao) => {
    $$("form[data-api]", cartao).forEach((form) => {
        form.addEventListener("submit", (evento) => {
            evento.preventDefault();
            executar($('button[type="submit"]', form), async () => {
                const dados = Object.fromEntries(new FormData(form).entries());
                const r = await postJson(form.dataset.api, dados);
                const caixa = criarResultado(cartao);
                if (form.dataset.api.includes("euler")) {
                    kpis(caixa, [
                        { rotulo: "Carga crítica Pcr", valor: r.carga_critica_kn },
                        { rotulo: "Pcr", valor: r.carga_critica },
                        { rotulo: "Carga de projeto", valor: r.projeto },
                    ]);
                } else {
                    kpis(caixa, [
                        { rotulo: "Trinca crítica a_c", valor: r.comprimento_critico },
                        { rotulo: "K_I em a_c/4", valor: r.ki_no_quarto },
                    ]);
                }
                nota(caixa, r.detalhe);
            });
        });
    });
});

/* ---------- Materiais ---------- */

registrar("materiais", (cartao) => {
    const form = $("form", cartao);
    form.addEventListener("submit", (evento) => {
        evento.preventDefault();
        executar($("button", form), async () => {
            const dados = Object.fromEntries(new FormData(form).entries());
            const r = await postJson("/api/material/laminado", dados);
            const caixa = criarResultado(cartao);
            nota(caixa, r.detalhe);
            const resumo = r.resumo || {};
            const tabela = document.createElement("table");
            tabela.className = "tabela";
            tabela.innerHTML = "<tr><th>Grandeza CLT</th><th>Valor</th></tr>";
            const nomes = { A: "Matriz de extensão A (N/m)", B: "Matriz de acoplamento B (N)", D: "Matriz de flexão D (N·m)",
                            deformacao_meio: "Deformação no plano médio ε₀", curvatura: "Curvatura κ (1/m²)",
                            resultantes_termicas: "Resultantes térmicas" };
            for (const chave of ["A", "B", "D", "deformacao_meio", "curvatura"]) {
                if (!(chave in resumo)) continue;
                const linha = document.createElement("tr");
                linha.innerHTML = `<td style="font-family:var(--font);color:var(--text-dim)">${nomes[chave] || chave}</td><td>${escutaHtml(JSON.stringify(resumo[chave]))}</td>`;
                tabela.appendChild(linha);
            }
            if (resumo.pilhas && Array.isArray(resumo.pilhas)) {
                resumo.pilhas.forEach((pilha, i) => {
                    const sup = pilha.superior || {};
                    const linha = document.createElement("tr");
                    linha.innerHTML = `<td style="font-family:var(--font);color:var(--text-dim)">σ local camada ${i + 1} (topo)</td><td>${escutaHtml(JSON.stringify(sup.tensao_local ?? []))}</td>`;
                    tabela.appendChild(linha);
                });
            }
            caixa.appendChild(tabela);
        });
    });
});

/* ---------- Matrizes ---------- */

registrar("matrizes", (cartao) => {
    $$("form[data-api]", cartao).forEach((form) => {
        form.addEventListener("submit", (evento) => {
            evento.preventDefault();
            executar($('button[type="submit"]', form), async () => {
                const dados = Object.fromEntries(new FormData(form).entries());
                const r = await postJson(form.dataset.api, dados);
                const caixa = criarResultado(cartao);
                if (r.det !== undefined) kpis(caixa, [{ rotulo: "det(A)", valor: r.det }]);
                if (r.inversa) {
                    const tabela = document.createElement("table");
                    tabela.className = "tabela";
                    tabela.innerHTML = "<tr><th>A⁻¹</th></tr>" + r.inversa.map((l) => `<tr><td>[${l.join(", ")}]</td></tr>`).join("");
                    caixa.appendChild(tabela);
                }
                if (r.solucao) {
                    kpis(caixa, [{ rotulo: "Solução x", valor: "[" + r.solucao.join(", ") + "]" },
                                 { rotulo: "Residual ‖Ax−b‖", valor: r.residual }]);
                }
            });
        });
    });
});

/* ---------- Constantes ---------- */

registrar("constantes", (cartao) => {
    const lista = $(".chips", cartao);
    getJson("/api/constantes").then((r) => {
        lista.innerHTML = "";
        r.constantes.forEach((c) => {
            const chip = document.createElement("button");
            chip.className = "chip";
            chip.type = "button";
            chip.innerHTML = `${escutaHtml(c.nome)} <small>${escutaHtml(c.valor)}</small>`;
            chip.title = "clique para copiar";
            chip.addEventListener("click", async () => {
                try { await navigator.clipboard.writeText(c.valor); }
                catch { /* clipboard indisponível */ }
                chip.classList.add("ativo");
                setTimeout(() => chip.classList.remove("ativo"), 900);
            });
            lista.appendChild(chip);
        });
    }).catch((erro) => { lista.innerHTML = `<div class="erro-caixa">⚠️ ${escutaHtml(erro.message)}</div>`; });
});

/* ---------- Tabela periódica ---------- */

registrar("elementos", (cartao) => {
    const entrada = $('input[name="q"]', cartao);
    const resultados = $(".resultados-elementos", cartao);
    let timer;
    async function buscar() {
        try {
            const r = await getJson("/api/elementos/buscar?q=" + encodeURIComponent(entrada.value));
            resultados.innerHTML = "";
            const tabela = document.createElement("table");
            tabela.className = "tabela";
            tabela.innerHTML = "<tr><th>Símbolo</th><th>Nome</th><th>Z</th><th>Massa (u)</th></tr>";
            r.resultados.forEach((el) => {
                const linha = document.createElement("tr");
                linha.style.cursor = "pointer";
                linha.innerHTML = `<td style="color:var(--accent);font-weight:700">${escutaHtml(el.simbolo)}</td>
                                   <td style="font-family:var(--font)">${escutaHtml(el.nome)}</td>
                                   <td>${escutaHtml(el.numero_atomico ?? "")}</td>
                                   <td>${escutaHtml(el.massa ?? "")}</td>`;
                linha.addEventListener("click", () => detalhe(el.simbolo));
                tabela.appendChild(linha);
            });
            resultados.appendChild(tabela);
        } catch (erro) {
            resultados.innerHTML = `<div class="erro-caixa">⚠️ ${escutaHtml(erro.message)}</div>`;
        }
    }
    async function detalhe(simbolo) {
        try {
            const r = await getJson("/api/elementos/detalhe/" + encodeURIComponent(simbolo));
            resultados.innerHTML = "";
            const caixa = document.createElement("div");
            caixa.className = "kpis";
            const chavesInteressantes = ["name", "atomic_number", "atomic_weight", "density_g_cm3",
                "melting_point_K", "boiling_point_K", "electronegativity_pauling",
                "thermal_conductivity_W_mK", "specific_heat_J_gK", "electron_configuration"];
            for (const chave of chavesInteressantes) {
                if (r.dados[chave] === undefined) continue;
                const div = document.createElement("div");
                div.className = "kpi";
                div.innerHTML = `<div class="rotulo">${escutaHtml(chave.replace(/_/g, " "))}</div>
                                 <div class="valor pequeno">${escutaHtml(String(r.dados[chave]))}</div>`;
                caixa.appendChild(div);
            }
            resultados.appendChild(caixa);
        } catch (erro) {
            resultados.innerHTML = `<div class="erro-caixa">⚠️ ${escutaHtml(erro.message)}</div>`;
        }
    }
    entrada.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(buscar, 250); });
    buscar();
});

/* ---------- Venturi 3D ---------- */

registrar("venturi", (cartao) => {
    const form = $("form", cartao);
    const variavel = $("#variavel-venturi", cartao);
    let ultimo = null;

    function redesenhar() {
        if (!ultimo) return;
        const tela = $(".grafico-venturi canvas", cartao);
        if (!tela) return;
        const campo = ultimo.perfis[variavel.value];
        desenharDispersao(tela, ultimo.pontos, {
            contorno: ultimo.geometria,
            comprimento: ultimo.comprimento,
            rotuloEscala: { mach: "Mach", pressao_kpa: "Pressão (kPa)", temperatura: "T (K)", densidade: "ρ (kg/m³)" }[variavel.value],
        });
    }

    variavel.addEventListener("change", redesenhar);

    form.addEventListener("submit", (evento) => {
        evento.preventDefault();
        executar($("button", form), async () => {
            const dados = Object.fromEntries(new FormData(form).entries());
            const r = await postJson("/api/venturi", dados);
            const caixa = criarResultado(cartao);
            kpis(caixa, [
                { rotulo: "Regime", valor: r.regime },
                { rotulo: "Mach na garganta", valor: r.saida.Mach_garganta },
                { rotulo: "Mach na saída", valor: r.saida.Mach_saida },
                { rotulo: "P saída", valor: r.saida.Pressao_saida_MPa + " MPa" },
                { rotulo: "T saída", valor: r.saida.Temperatura_saida_K + " K" },
            ]);
            /* campos fora do formulário (controle de visualização) */
            const env = $(".linha-acoes", cartao);
            if (!$("#variavel-venturi", env)) {
                const seletor = document.createElement("select");
                seletor.id = "variavel-venturi";
                seletor.style.width = "auto";
                seletor.innerHTML = `<option value="mach">Mach</option><option value="pressao_kpa">Pressão</option>
                                      <option value="temperatura">Temperatura</option><option value="densidade">Densidade</option>`;
                const rotulo = document.createElement("span");
                rotulo.className = "dica";
                rotulo.textContent = "mapa do bocal:";
                env.append(rotulo, seletor);
                seletor.addEventListener("change", redesenhar);
            }
            /* pontos do campo (plano x-y, várias fracções radiais) */
            const pontos = [];
            const nCamadas = 9;
            for (let i = 0; i < r.eixo_x.length; i += 4) {
                const x = r.eixo_x[i];
                const raio = r.geometria.superior[r.geometria.x.indexOf(x)] ?? 0;
                for (let j = 0; j < nCamadas; j++) {
                    const fracao = j / (nCamadas - 1);
                    const y = raio * (2 * fracao - 1);
                    pontos.push({ x, y, v: r.perfis[variavel ? variavel.value : "mach"][i] });
                }
            }
            ultimo = { perfis: r.perfis, geometria: r.geometria, comprimento: r.eixo_x[r.eixo_x.length - 1], pontos };

            const envolucro = document.createElement("div");
            envolucro.className = "grafico-caixa grafico-venturi";
            const tela = document.createElement("canvas");
            envolucro.appendChild(tela);
            caixa.appendChild(envolucro);
            desenharDispersao(tela, ultimo.pontos, {
                contorno: ultimo.geometria,
                comprimento: ultimo.comprimento,
                rotuloEscala: "Mach",
            });
            grafico(caixa, [
                { x: r.eixo_x, y: r.perfis.mach, rotulo: "Mach (eixo)", cor: "#7c9aff" },
                { x: r.eixo_x, y: r.perfis.pressao_kpa.map((p) => p / Math.max(...r.perfis.pressao_kpa)), rotulo: "P/P₀", cor: "#3fd6c6" },
            ]);
        });
    });
});

/* ---------------- inicialização ---------------- */

document.addEventListener("DOMContentLoaded", () => {
    $$(".ferramenta").forEach((cartao) => {
        const cabecalho = $(".cabecalho-ferramenta", cartao);
        cabecalho.addEventListener("click", () => {
            cartao.classList.toggle("aberta");
            if (cartao.classList.contains("aberta") && !cartao.dataset.inicializado) {
                cartao.dataset.inicializado = "1";
                const init = FERRAMENTAS[cartao.id];
                if (init) init(cartao);
            }
        });
    });

    /* sub-abas */
    $$(".abas .aba").forEach((aba) => {
        aba.addEventListener("click", () => {
            const painelPai = aba.closest(".painel-aba, .corpo-ferramenta");
            const irmaos = $$(".aba", aba.parentElement);
            irmaos.forEach((a) => a.classList.remove("ativa"));
            aba.classList.add("ativa");
            const alvo = aba.dataset.aba;
            $$("[data-painel]", painelPai.closest(".corpo-ferramenta")).forEach((p) => {
                p.classList.toggle("ativa", p.dataset.painel === alvo);
            });
        });
    });

    /* abrir a primeira ferramenta automaticamente */
    const primeira = $(".ferramenta");
    if (primeira) {
        primeira.classList.add("aberta");
        if (!primeira.dataset.inicializado) {
            primeira.dataset.inicializado = "1";
            const init = FERRAMENTAS[primeira.id];
            if (init) init(primeira);
        }
    }
});
