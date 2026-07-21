# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nemo_curator.stages.text.filters.doc_filter import DocumentFilter
from .kenlm_utility import KenlmModel

class PerplexityFilter(DocumentFilter):
    """
    Filters documents based on KenLM n-gram language model perplexity.

    Documents whose perplexity falls outside [min_perplexity, max_perplexity]
    are discarded. Lower perplexity indicates text that more closely resembles
    the training corpus of the language model (typically high-quality web text).

    """

    def __init__(
        self,
        model_path: str,
        min_perplexity: float,
        max_perplexity: float,
        lang: str = "en",
        lower_case: bool = False,
        remove_accents: bool = False,
        normalize_numbers: bool = True,
        punctuation: int = 1,
    ) -> None:
        """
        Args:
            model_path (str): Path to the directory containing the KenLM binary
                model ({lang}.arpa.bin) and SentencePiece model ({lang}.sp.model).
            min_perplexity (float): Minimum perplexity threshold (inclusive). Documents
                with perplexity below this value are discarded.
            max_perplexity (float): Maximum perplexity threshold (inclusive). Documents
                with perplexity above this value are discarded.
            lang (str): Language code used to locate the model files. Defaults to "en".
            lower_case (bool): Whether to lowercase text before scoring. Defaults to False.
            remove_accents (bool): Whether to strip accent characters before scoring. Defaults to False.
            normalize_numbers (bool): Whether to replace digits with 0 before scoring. Defaults to True.
            punctuation (int): Punctuation handling mode — 1 replaces Unicode punctuation
                with ASCII equivalents, 2 removes it entirely, 0 disables. Defaults to 1.
        """
        super().__init__()
        self.model = KenlmModel(
            model_path=model_path,
            language=lang,
            lower_case=lower_case,
            remove_accents=remove_accents,
            normalize_numbers=normalize_numbers,
            punctuation=punctuation,
        )
        self.min_perplexity = min_perplexity
        self.max_perplexity = max_perplexity

    def score_document(self, text: str) -> float:
        """
        Compute the KenLM perplexity score for the given text.

        Args:
            text (str): The document text to score.

        Returns:
            float: The perplexity score. Lower values indicate text more similar
                to the language model's training corpus.
        """
        return self.model.get_perplexity(text, normalize=True)

    def keep_document(self, score: float) -> bool:
        """
        Determine whether to keep a document based on its perplexity score.

        Args:
            score (float): The perplexity score returned by score_document().

        Returns:
            bool: True if the score is within [min_perplexity, max_perplexity], False otherwise.
        """
        return self.min_perplexity <= score <= self.max_perplexity