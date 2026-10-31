"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type Status = { kind: "ok"; text: string } | { kind: "error"; text: string };

type JobResult = { documents_ingested?: number; files_found?: number } | null;

type JobListItem = {
  job_id: string;
  kind: string;
  source: string | null;
  status: string;
  result: JobResult;
  error: string | null;
  created_at: string;
  updated_at: string;
};

const POLL_INTERVAL_MS = 1000;
const POLL_TIMEOUT_MS = 5 * 60 * 1000;
const JOB_LIST_POLL_MS = 3000;

function formatJobResult(result: JobResult): string {
  if (!result) return "";
  return "files_found" in result
    ? `${result.documents_ingested} documents ingested, ${result.files_found} files found.`
    : `${result.documents_ingested} documents ingested.`;
}

export async function pollJob(jobId: string): Promise<Status> {
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const res = await fetch(`/api/jobs/${jobId}`);
    const data = await res.json();
    if (!res.ok) {
      return { kind: "error", text: data.error ?? "Job lookup failed." };
    }
    if (data.status === "done") {
      const r = data.result ?? {};
      return {
        kind: "ok",
        text:
          "files_found" in r
            ? `${r.documents_ingested} documents ingested, ${r.files_found} files found.`
            : `${r.documents_ingested} documents ingested.`,
      };
    }
    if (data.status === "failed") {
      return { kind: "error", text: data.error ?? "Ingestion failed." };
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
  return { kind: "error", text: "Timed out waiting for ingestion to finish." };
}

export default function IngestPage() {
  const [source, setSource] = useState("");
  const [loading, setLoading] = useState(false);
  const [sourceStatus, setSourceStatus] = useState<Status | null>(null);
  const [reindexLoading, setReindexLoading] = useState(false);
  const [reindexStatus, setReindexStatus] = useState<Status | null>(null);
  const [jobs, setJobs] = useState<JobListItem[]>([]);

  const canIngest = source.trim().length > 0 && !loading;

  useEffect(() => {
    let cancelled = false;
    let interval: ReturnType<typeof setInterval> | null = null;

    async function tick() {
      try {
        const res = await fetch("/api/jobs");
        if (!res.ok || cancelled) return;
        const data = await res.json();
        const list: JobListItem[] = data.jobs ?? [];
        if (cancelled) return;
        setJobs(list);
        const inFlight = list.some((job) => job.status === "queued" || job.status === "processing");
        if (!inFlight && interval) {
          clearInterval(interval);
          interval = null;
        }
      } catch {
        // Keep showing the last known list rather than surfacing a loud error here.
      }
    }

    tick();
    interval = setInterval(tick, JOB_LIST_POLL_MS);

    return () => {
      cancelled = true;
      if (interval) clearInterval(interval);
    };
  }, []);

  async function ingestSource() {
    const value = source.trim();
    if (!value) return;

    setLoading(true);
    setSourceStatus(null);
    try {
      const res = await fetch("/api/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: value }),
      });
      const data = await res.json();
      if (!res.ok) {
        setSourceStatus({ kind: "error", text: data.error });
      } else {
        const result = await pollJob(data.job_id);
        setSourceStatus(result);
        if (result.kind === "ok") setSource("");
      }
    } catch {
      setSourceStatus({ kind: "error", text: "Couldn't reach the server — is it running?" });
    } finally {
      setLoading(false);
    }
  }

  async function reindexAll() {
    setReindexLoading(true);
    setReindexStatus(null);
    try {
      const res = await fetch("/api/index", { method: "POST" });
      const data = await res.json();
      if (!res.ok) {
        setReindexStatus({ kind: "error", text: data.error });
      } else {
        setReindexStatus(await pollJob(data.job_id));
      }
    } catch {
      setReindexStatus({ kind: "error", text: "Couldn't reach the server — is it running?" });
    } finally {
      setReindexLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="masthead">
        <h1>Add documents</h1>
        <p>Ingest a source, or re-index everything already in the corpus.</p>
        <nav>
          <Link href="/">Back to chat</Link>
        </nav>
      </header>
      <section className="ingest-section">
        <h2>Ingest a source</h2>
        <p className="ingest-section__hint">
          A URL, or a file path under <code>data/documents/</code>.
        </p>
        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            if (canIngest) ingestSource();
          }}
        >
          <input
            type="text"
            aria-label="Source"
            placeholder="https://example.com/paper.pdf"
            value={source}
            onChange={(e) => setSource(e.target.value)}
          />
          <button type="submit" disabled={!canIngest}>
            Ingest
          </button>
        </form>
        {loading && <p className="status status--pending">Ingesting… this can take a while for a large document.</p>}
        {!loading && sourceStatus && (
          <p
            className={sourceStatus.kind === "error" ? "status status--error" : "status"}
            role={sourceStatus.kind === "error" ? "alert" : undefined}
          >
            {sourceStatus.kind === "error" ? `Error: ${sourceStatus.text}` : sourceStatus.text}
          </p>
        )}
      </section>
      <section className="ingest-section">
        <h2>Re-index everything</h2>
        <p className="ingest-section__hint">
          Scans every file already under <code>data/documents/</code> and re-ingests it.
        </p>
        <button type="button" disabled={reindexLoading} onClick={reindexAll}>
          Re-index everything
        </button>
        {reindexLoading && <p className="status status--pending">Re-indexing… this can take a while.</p>}
        {!reindexLoading && reindexStatus && (
          <p
            className={reindexStatus.kind === "error" ? "status status--error" : "status"}
            role={reindexStatus.kind === "error" ? "alert" : undefined}
          >
            {reindexStatus.kind === "error" ? `Error: ${reindexStatus.text}` : reindexStatus.text}
          </p>
        )}
      </section>
      <section className="ingest-section">
        <h2>Recent jobs</h2>
        {jobs.length === 0 ? (
          <p className="ingest-section__hint">No ingestion jobs yet.</p>
        ) : (
          <ul className="job-list">
            {jobs.map((job) => (
              <li key={job.job_id} className="job-list__item">
                <div className="job-list__row">
                  <span className="job-list__source">{job.source ?? job.kind}</span>
                  <span className={`status-badge status-badge--${job.status}`}>{job.status}</span>
                </div>
                <div className="job-list__meta">
                  <span className="job-list__time">{new Date(job.created_at).toLocaleString()}</span>
                  {job.status === "done" && job.result && <span>{formatJobResult(job.result)}</span>}
                  {job.status === "failed" && job.error && (
                    <span className="status status--error">{job.error}</span>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
