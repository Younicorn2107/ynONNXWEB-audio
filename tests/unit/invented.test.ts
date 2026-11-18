import { describe, expect, it } from "vitest";

import { isCreditLine, isInventedLine, isNonSpeechLine } from "../../src/whisper/invented.js";

describe("the subtitle credits Whisper invents", () => {
  it.each([
    "Sous-titrage Société Radio-Canada",
    " Sous-titres réalisés par la communauté d'Amara.org",
    "Sous-titrage MFP.",
    "Sous-titrage ST' 501",
    "Subtitles by the Amara.org community",
    "Untertitel im Auftrag des ZDF, 2021",
    "Subtítulos realizados por la comunidad de Amara.org",
    "Sottotitoli creati dalla comunità Amara.org",
  ])("drops %j", (text) => {
    expect(isCreditLine(text)).toBe(true);
    expect(isInventedLine(text)).toBe(true);
  });

  it.each([
    "Merci.",
    "Applaudissements.",
    "Le sous-titrage aide les sourds et les malentendants.",
    "C'est un des premiers à se faire enregistrer.",
    "Radio-Canada a diffusé le débat hier soir.",
    "",

    "On parle souvent de la communauté d'Amara.org, qui traduit des vidéos " +
      "bénévolement depuis des années dans plus de cent langues différentes",
  ])("keeps %j", (text) => {
    expect(isCreditLine(text)).toBe(false);
  });

  it("reads a curly apostrophe as a straight one", () => {
    expect(isCreditLine("Sous-titrage ST’ 501")).toBe(true);
  });
});

describe("a line with no word", () => {
  it.each(["...", "- -", " . ", "*cris*", "[Musique]", "(Applaudissements)", "♪ ♪"])("drops %j", (text) => {
    expect(isNonSpeechLine(text)).toBe(true);
    expect(isInventedLine(text, "fr")).toBe(true);
  });

  it.each(["Merci.", "Il a dit *non* hier.", "Bonjour (rires) toi", "42", ""])("keeps %j", (text) => {
    expect(isNonSpeechLine(text)).toBe(false);
  });

  it("counts letters of every script as words", () => {
    expect(isInventedLine("Спасибо.")).toBe(false);
    expect(isInventedLine("谢谢。")).toBe(false);
  });
});
