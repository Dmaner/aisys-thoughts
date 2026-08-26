/* DOM rendering for homepage cards, blog rows, article pages, and theme state. */
(function initViews(global) {
  "use strict";

  const app = (global.MLSystem = global.MLSystem || {});
  const { escapeHtml, formatDate, sortByCreated, uniqueTags } = app.utils;
  const { parsePostMetadata } = app.content;
  const { markdownToHtml } = app.markdown;

  const nodes = {
    root: document.documentElement,
    themeToggle: document.querySelector("[data-theme-toggle]"),
    currentList: document.querySelector("[data-current-list]"),
    blogList: document.querySelector("[data-blog-list]"),
    blogIndexShell: document.querySelector("[data-blog-index-shell]"),
    projectsShell: document.querySelector("[data-projects-shell]"),
    projectTimeline: document.querySelector("[data-project-timeline]"),
    articleShell: document.querySelector("[data-article-shell]"),
    articleToc: document.querySelector("[data-article-toc]"),
    markdownBody: document.querySelector("[data-markdown-body]"),
    articleMeta: document.querySelector("[data-article-meta]"),
    backHome: document.querySelector("[data-back-home]"),
  };
  let articleTocCleanup = null;
  let articleRenderToken = 0;

  /* Applies saved light/dark preference and persists future theme changes. */
  function initTheme() {
    const storedTheme = localStorage.getItem("ml-system-theme");
    if (storedTheme) nodes.root.dataset.theme = storedTheme;

    nodes.themeToggle?.addEventListener("click", () => {
      const nextTheme = nodes.root.dataset.theme === "dark" ? "light" : "dark";
      nodes.root.dataset.theme = nextTheme;
      localStorage.setItem("ml-system-theme", nextTheme);
    });
  }

  /* Renders reusable tag pills for cards, blog rows, article pages, and search. */
  function renderTagList(tags = []) {
    const cleanTags = uniqueTags(tags);
    if (!cleanTags.length) return "";
    return `<div class="tag-list">${cleanTags
      .map((tag) => `<span class="tag">#${escapeHtml(tag)}</span>`)
      .join("")}</div>`;
  }

  function setHomeSectionsHidden(hidden) {
    document.querySelectorAll(".section-block").forEach((section) => {
      section.hidden = hidden;
    });
  }

  function clearArticleToc() {
    articleTocCleanup?.();
    articleTocCleanup = null;
    nodes.articleShell.classList.remove("has-article-toc");
    nodes.articleToc.hidden = true;
    nodes.articleToc.replaceChildren();
  }

  function renderArticleToc() {
    clearArticleToc();

    const headings = Array.from(nodes.markdownBody.querySelectorAll("h2, h3"));
    if (!headings.length) return;

    const tocList = document.createElement("nav");
    tocList.className = "article-toc-list";
    tocList.setAttribute("aria-label", "Article sections");

    const buttons = headings.map((heading, index) => {
      const headingId = `article-section-${index + 1}`;
      const level = heading.tagName.toLowerCase();
      const button = document.createElement("button");

      heading.id = headingId;
      heading.classList.add("article-section-heading");
      button.type = "button";
      button.className = `article-toc-link article-toc-link--${level}`;
      button.textContent = heading.textContent.trim();
      button.setAttribute("aria-controls", headingId);
      button.addEventListener("click", () => {
        const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        heading.scrollIntoView({
          behavior: reducedMotion ? "auto" : "smooth",
          block: "start",
        });
      });
      tocList.append(button);
      return button;
    });

    nodes.articleToc.append(tocList);
    nodes.articleToc.hidden = false;
    nodes.articleShell.classList.add("has-article-toc");

    let activeIndex = -1;
    let animationFrame = null;

    function keepActiveButtonVisible(button) {
      if (!button || tocList.clientHeight === 0) return;

      const viewTop = tocList.scrollTop;
      const viewBottom = viewTop + tocList.clientHeight;
      const buttonTop = button.offsetTop;
      const buttonBottom = buttonTop + button.offsetHeight;

      if (buttonTop < viewTop) tocList.scrollTop = buttonTop;
      else if (buttonBottom > viewBottom) {
        tocList.scrollTop = buttonBottom - tocList.clientHeight;
      }
    }

    function setActiveHeading(nextIndex) {
      if (nextIndex === activeIndex) {
        keepActiveButtonVisible(buttons[nextIndex]);
        return;
      }
      activeIndex = nextIndex;

      buttons.forEach((button, index) => {
        const isActive = index === activeIndex;
        button.classList.toggle("is-active", isActive);
        if (isActive) button.setAttribute("aria-current", "location");
        else button.removeAttribute("aria-current");
      });

      keepActiveButtonVisible(buttons[activeIndex]);
    }

    function updateActiveHeading() {
      animationFrame = null;
      const activationLine = Math.min(96, window.innerHeight * 0.12);
      let nextIndex = 0;

      const pageBottom = window.scrollY + window.innerHeight;
      const documentBottom = document.documentElement.scrollHeight;
      const hasScrollablePage = documentBottom - window.innerHeight > 2;
      if (hasScrollablePage && pageBottom >= documentBottom - 2) {
        nextIndex = headings.length - 1;
      } else {
        headings.forEach((heading, index) => {
          if (heading.getBoundingClientRect().top <= activationLine) nextIndex = index;
        });
      }
      setActiveHeading(nextIndex);
    }

    function scheduleActiveHeadingUpdate() {
      if (animationFrame !== null) return;
      animationFrame = window.requestAnimationFrame(updateActiveHeading);
    }

    window.addEventListener("scroll", scheduleActiveHeadingUpdate, { passive: true });
    window.addEventListener("resize", scheduleActiveHeadingUpdate);
    updateActiveHeading();

    articleTocCleanup = () => {
      if (animationFrame !== null) window.cancelAnimationFrame(animationFrame);
      window.removeEventListener("scroll", scheduleActiveHeadingUpdate);
      window.removeEventListener("resize", scheduleActiveHeadingUpdate);
    };
  }

  function showHome() {
    articleRenderToken += 1;
    clearArticleToc();
    nodes.articleShell.hidden = true;
    nodes.blogIndexShell.hidden = true;
    nodes.projectsShell.hidden = true;
    setHomeSectionsHidden(false);
  }

  function showBlogIndex() {
    articleRenderToken += 1;
    clearArticleToc();
    nodes.articleShell.hidden = true;
    nodes.blogIndexShell.hidden = false;
    nodes.projectsShell.hidden = true;
    setHomeSectionsHidden(true);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function showProjects() {
    articleRenderToken += 1;
    clearArticleToc();
    nodes.articleShell.hidden = true;
    nodes.blogIndexShell.hidden = true;
    nodes.projectsShell.hidden = false;
    setHomeSectionsHidden(true);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function renderArticleMeta(item, tags = item.tags) {
    nodes.articleMeta.innerHTML = `
      <div class="article-topline">
        <span>${escapeHtml(item.category)}</span>
        <span>${escapeHtml(formatDate(item.createdAt))}</span>
      </div>
      <h1>${escapeHtml(item.title)}</h1>
      <p>${escapeHtml(item.description)}</p>
      ${renderTagList(tags)}
    `;
  }

  /* Opens a single post and renders embedded Markdown content into the article shell. */
  async function openPost(slug, contentIndex) {
    const item = contentIndex.find((entry) => entry.slug === slug);
    if (!item) {
      showHome();
      return;
    }
    const renderToken = ++articleRenderToken;

    setHomeSectionsHidden(true);
    nodes.blogIndexShell.hidden = true;
    nodes.projectsShell.hidden = true;
    nodes.articleShell.hidden = false;
    clearArticleToc();
    renderArticleMeta(item);
    nodes.markdownBody.innerHTML = "<p>Loading Markdown...</p>";

    if (typeof item.markdown === "string") {
      const metadata = parsePostMetadata(item.markdown);
      const tags = uniqueTags(metadata.tags.length ? metadata.tags : item.tags || []);
      const description = metadata.desc || item.description;
      renderArticleMeta({ ...item, description, tags }, tags);
      nodes.markdownBody.innerHTML = item.markdown.trim()
        ? markdownToHtml(item.markdown, { title: item.title })
        : `<p>This Markdown file is currently empty.</p>`;
      renderArticleToc();
      window.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }

    try {
      const response = await fetch(item.path);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const markdown = await response.text();
      if (renderToken !== articleRenderToken) return;
      const metadata = parsePostMetadata(markdown);
      const tags = uniqueTags(metadata.tags.length ? metadata.tags : item.tags || []);
      const description = metadata.desc || item.description;
      renderArticleMeta({ ...item, description, tags }, tags);
      nodes.markdownBody.innerHTML = markdown.trim()
        ? markdownToHtml(markdown, { title: item.title })
        : `<p>This Markdown file is currently empty.</p><p><a href="${escapeHtml(item.path)}">Open raw Markdown</a></p>`;
      renderArticleToc();
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch {
      if (renderToken !== articleRenderToken) return;
      clearArticleToc();
      nodes.markdownBody.innerHTML = `
        <p>This post is missing embedded Markdown content.</p>
        <p>Run <code>node website/tools/embed-markdown.mjs</code> from the repository root, then refresh this page.</p>
        <p><a href="${escapeHtml(item.path)}">Open raw Markdown</a></p>
      `;
    }
  }

  function renderBlogCard(item) {
    return `
      <a class="window-card topic-card" href="#post=${escapeHtml(item.slug)}" data-searchable data-slug="${escapeHtml(item.slug)}">
        <div class="window-dots" aria-hidden="true">
          <span></span>
          <span></span>
          <span></span>
        </div>
        <p class="kicker">${escapeHtml(item.category)}</p>
        <h3>${escapeHtml(item.title)}</h3>
        <p>${escapeHtml(item.description)}</p>
        ${renderTagList(item.tags)}
        <div class="blue-stroke"></div>
      </a>
    `;
  }

  function renderBlogRow(item) {
    return `
      <a class="window-card blog-row" href="#post=${escapeHtml(item.slug)}" data-searchable data-slug="${escapeHtml(item.slug)}">
        <div>
          <p class="kicker">${escapeHtml(item.category)}</p>
          <h3>${escapeHtml(item.title)}</h3>
          <p>${escapeHtml(item.description)}</p>
          ${renderTagList(item.tags)}
        </div>
        <time datetime="${escapeHtml(item.createdAt || "")}">${escapeHtml(formatDate(item.createdAt))}</time>
      </a>
    `;
  }

  /* Current shows the six latest created posts, producing a 2 by 3 grid on desktop. */
  function renderCurrentCards(contentIndex) {
    const cards = sortByCreated(contentIndex).slice(0, 6);
    nodes.currentList.innerHTML = cards.map(renderBlogCard).join("");
  }

  function renderBlogIndex(contentIndex) {
    nodes.blogList.innerHTML = sortByCreated(contentIndex).map(renderBlogRow).join("");
  }

  function formatProjectMonth(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return { year: "", month: "" };
    return {
      year: String(date.getFullYear()),
      month: String(date.getMonth() + 1).padStart(2, "0"),
    };
  }

  function renderProjectItem(item) {
    const date = formatProjectMonth(item.createdAt);
    const actions = [
      item.href
        ? `<a class="project-action" href="${escapeHtml(item.href)}" target="_blank" rel="noreferrer" aria-label="Open GitHub repository for ${escapeHtml(item.title)}">
            <svg aria-hidden="true" viewBox="0 0 24 24">
              <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3.2-.4 6.5-1.6 6.5-7A5.4 5.4 0 0 0 19 3.8 5 5 0 0 0 18.9 0s-1.2-.4-4 1.5a13.7 13.7 0 0 0-7.3 0C4.8-.4 3.6 0 3.6 0a5 5 0 0 0-.1 3.8A5.4 5.4 0 0 0 2 7.5c0 5.4 3.3 6.6 6.5 7a4.8 4.8 0 0 0-1 3.5v4" />
              <path d="M8 19c-3 .9-3-1.5-4-2" />
            </svg>
          </a>`
        : "",
      item.homepage
        ? `<a class="project-action" href="${escapeHtml(item.homepage)}" target="_blank" rel="noreferrer" aria-label="Open project page for ${escapeHtml(item.title)}">
            <svg aria-hidden="true" viewBox="0 0 24 24">
              <path d="M7 17 17 7" />
              <path d="M9 7h8v8" />
            </svg>
          </a>`
        : "",
    ].filter(Boolean);
    const language = item.language
      ? `<div class="project-language-row">
          <span class="project-language-chip">
            <span aria-hidden="true"></span>
            ${escapeHtml(item.language)}
          </span>
        </div>`
      : "";

    return `
      <article class="project-timeline-item">
        <time class="project-date" datetime="${escapeHtml(item.createdAt || "")}">
          <span>${escapeHtml(date.year)}</span>
          <strong>${escapeHtml(date.month)}</strong>
        </time>
        <div class="window-card project-card">
          <div class="window-dots" aria-hidden="true">
            <span></span>
            <span></span>
            <span></span>
          </div>
          <p class="kicker">${escapeHtml(item.repo)}</p>
          <div class="project-title-row">
            <h3>${escapeHtml(item.title)}</h3>
            ${actions.length ? `<div class="project-actions">${actions.join("")}</div>` : ""}
          </div>
          <p>${escapeHtml(item.description)}</p>
          ${language}
          ${renderTagList(item.tags)}
          <div class="blue-stroke"></div>
        </div>
      </article>
    `;
  }

  function renderProjectTimeline(projects = []) {
    const sortedProjects = sortByCreated(projects);
    nodes.projectTimeline.innerHTML = sortedProjects.length
      ? sortedProjects.map(renderProjectItem).join("")
      : `<p class="empty-state">No projects yet.</p>`;
  }

  app.views = {
    initTheme,
    nodes,
    openPost,
    renderBlogIndex,
    renderCurrentCards,
    renderProjectTimeline,
    renderTagList,
    showBlogIndex,
    showHome,
    showProjects,
  };
})(window);
