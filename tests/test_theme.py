"""Backdrop colour, from the dropdown value to the words the generator reads.

The six named studio sweeps are one setting spread across four files: an enum
member, a hex in a lookup table, a phrase template, and a generated TypeScript
constant the wizard paints a swatch from.  Nothing links them at runtime, so
each of these tests holds one joint in that chain closed.

Two of them are about a specific way this breaks quietly.  A new enum member
with no phrase raises ``KeyError`` inside the brief compiler — at job time, not
at import — and a phrase that qualifies itself after a comma survives the full
prompt style and reaches the compact one with its colour amputated.  Both
failures produce an ad on the wrong backdrop rather than an error.
"""

from __future__ import annotations

import adproviders as P
import pytest
from adschema import BackgroundTreatment, PromptStyle, ThemeSpec, sweep_colour_for
from adschema.request import HEX_COLOR
from adworker.briefs import _BACKGROUND_PHRASE, compile_briefs

#: The treatments whose name fixes their colour.
NAMED_SWEEPS = [b for b in BackgroundTreatment if sweep_colour_for(b) is not None]


def test_every_treatment_has_a_phrase():
    """Otherwise the compiler raises KeyError on a job, not on import."""
    missing = [b.value for b in BackgroundTreatment if b not in _BACKGROUND_PHRASE]
    assert not missing, f"backgrounds with no brief phrase: {missing}"


def test_every_sweep_colour_is_a_hex():
    for background in NAMED_SWEEPS:
        colour = sweep_colour_for(background)
        assert colour is not None
        assert HEX_COLOR.match(colour), f"{background.value} maps to {colour!r}"


def test_the_named_sweeps_are_the_ones_that_exist():
    """A guard on the set, not just the members.

    Renaming one is a UI-visible change and a wire-format change at once, so it
    should have to be made here too rather than passing silently.
    """
    assert [b.value for b in NAMED_SWEEPS] == [
        "studio_white",
        "studio_purple",
        "studio_blue",
        "studio_carbon_black",
        "studio_green",
        "studio_coral",
    ]


@pytest.mark.parametrize("background", NAMED_SWEEPS, ids=lambda b: b.value)
def test_a_named_sweep_resolves_its_own_colour(background):
    theme = ThemeSpec(background=background)
    assert theme.background_color == sweep_colour_for(background)


@pytest.mark.parametrize("background", NAMED_SWEEPS, ids=lambda b: b.value)
def test_a_named_sweep_overrides_a_colour_left_behind(background):
    """The label the user picked wins over a hex from an earlier choice.

    Switching from `seamless_color` to `studio_green` in the wizard leaves the old
    hex in the payload.  Honouring it would render a backdrop that contradicts the
    name printed next to it.
    """
    theme = ThemeSpec(background=background, background_color="#ce7777")
    assert theme.background_color == sweep_colour_for(background)


def test_studio_white_keeps_its_hex_out_of_the_prompt():
    """It resolves a colour like the others, but its phrase does not interpolate it.

    "white studio backdrop" already says white, so spending prompt tokens on
    "#ffffff" would buy nothing and would move ``prompt_sha256`` on every frame
    frozen from this backdrop.
    """
    assert sweep_colour_for(BackgroundTreatment.STUDIO_WHITE) == "#ffffff"
    assert "{colour}" not in _BACKGROUND_PHRASE[BackgroundTreatment.STUDIO_WHITE]


def test_seamless_color_still_falls_back_to_the_palette():
    """Unchanged behaviour, asserted because the validator that does it was rewritten."""
    theme = ThemeSpec(palette=["#2b3a55", "#ce7777"], background=BackgroundTreatment.SEAMLESS_COLOR)
    assert theme.background_color == "#2b3a55"

    bare = ThemeSpec(background=BackgroundTreatment.SEAMLESS_COLOR)
    assert bare.background_color == "#f2f2f2"


@pytest.mark.parametrize("style", list(PromptStyle), ids=lambda s: s.value)
@pytest.mark.parametrize(
    "background",
    [b for b in NAMED_SWEEPS if b is not BackgroundTreatment.STUDIO_WHITE],
    ids=lambda b: b.value,
)
async def test_the_sweep_colour_reaches_every_prompt(make_request, background, style):
    """Both prompt styles, because the compact one truncates at the first comma."""
    colour = sweep_colour_for(background)
    request = make_request(theme=ThemeSpec(background=background))
    briefs = await compile_briefs(request, P.MockLLMProvider(), style)

    assert briefs.briefs
    for brief in briefs.briefs:
        assert colour in brief.image_prompt, (
            f"{background.value} lost its colour in the {style.value} prompt"
        )
