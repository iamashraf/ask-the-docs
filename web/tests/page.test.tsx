import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import Page from "../app/page";

describe("Page", () => {
  test("renders the chat page heading", () => {
    render(<Page />);
    expect(screen.getByRole("heading")).toBeInTheDocument();
  });
});
