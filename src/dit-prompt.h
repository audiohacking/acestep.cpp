#pragma once
// dit-prompt.h: DiT text/lyric conditioning prompt strings.
//
// Builds the exact prompt text fed to the text encoder for the DiT's
// cross-attention conditioning. Shared by inference (src/pipeline-synth-ops.cpp,
// via build_prompt_strings) and dataset preparation (src/dit-prepare.h, Phase 4)
// so the two can never drift apart -- a LoRA trained here sees exactly the
// conditioning distribution ace-synth will feed it at inference time.
//
// Matches acestep-repo's core/generation/handler/prompt_utils.py
// (PromptMixin.build_dit_inputs / _format_lyrics), which is itself the
// original ACE-Step Python inference reference -- not
// acestep/training/dataset_builder_modules/preprocess_lyrics.py, whose
// encode_lyrics() tokenizes raw, unwrapped lyrics text. That divergence
// looks like a quirk/simplification specific to the community LoRA
// trainer's dataset builder; matching it would train the adapter on a
// lyric conditioning distribution ace-synth never actually produces. See
// TRAINING_DEV.md's Phase 4 section for the decision record.

#include <cstdio>
#include <string>

// text_out: "# Instruction\n{instruction}\n\n# Caption\n{caption}\n\n# Metas\n{metas}<|endoftext|>\n"
// lyric_out: "# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"
static void dit_build_prompt_strings(const std::string & instruction,
                                     const std::string & caption,
                                     const std::string & lyrics,
                                     int                 bpm,
                                     const std::string & keyscale,
                                     const std::string & timesignature,
                                     const std::string & vocal_language,
                                     float               duration,
                                     std::string &       text_out,
                                     std::string &       lyric_out) {
    char bpm_b[16] = "N/A";
    if (bpm > 0) {
        snprintf(bpm_b, sizeof(bpm_b), "%d", bpm);
    }
    const char * keyscale_b = keyscale.empty() ? "N/A" : keyscale.c_str();
    const char * timesig_b  = timesignature.empty() ? "N/A" : timesignature.c_str();
    const char * language_b = vocal_language.empty() ? "unknown" : vocal_language.c_str();

    char metas_b[512];
    snprintf(metas_b, sizeof(metas_b), "- bpm: %s\n- timesignature: %s\n- keyscale: %s\n- duration: %d seconds\n",
             bpm_b, timesig_b, keyscale_b, (int) duration);
    text_out = std::string("# Instruction\n") + instruction + "\n\n" + "# Caption\n" + caption + "\n\n" +
               "# Metas\n" + metas_b + "<|endoftext|>\n";
    lyric_out = std::string("# Languages\n") + language_b + "\n\n# Lyric\n" + lyrics + "<|endoftext|>";
}
