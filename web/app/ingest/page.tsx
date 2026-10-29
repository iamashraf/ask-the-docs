"use client";

import Link from "next/link";
import { useState } from "react";

type Status = { kind: "ok"; text: string } | { kind: "error"; text: string };

const POLL_INTERVAL_MS = 1000;
const POLL_TIMEOUT_MS = 5 * 60 * 1000;

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

  const canIngest = source.trim().length > 0 && !loading;

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
    </div>
  );
}
