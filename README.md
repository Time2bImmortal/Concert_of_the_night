# Concert_of_the_night

"""
A specialized audio analysis program for studying cricket chirping patterns. The program focuses on identifying 
and analyzing cricket syllables (individual pulses) and chirp sequences (groups of 3-4 syllables). Key features include:

1. Cricket Audio Processing:
   - Processes WAV recordings of cricket sounds
   - Detects individual syllables using amplitude threshold analysis
   - Identifies syllable start and end points
   - Extracts dominant frequencies of chirps
   - Applies signal smoothing and normalization for reliable detection

2. Chirp Pattern Analysis:
   - Groups syllables into characteristic cricket chirp patterns
   - Specifically analyzes 3-syllable and 4-syllable chirp sequences
   - Calculates timing parameters crucial for cricket communication:
     * Within-chirp timing (intervals between syllables in a chirp)
     * Between-chirp timing (intervals between consecutive chirps)
     * Transition patterns between different chirp types (3-3, 3-4, 4-3, 4-4)

3. Data Management:
   - Handles batch processing of multiple recordings
   - Creates annotated audio files marking syllable positions (using Audacity software)
   - Generates timing labels for detected chirps
   - Exports comprehensive analysis to CSV files including:
     * Syllable counts and timing
     * Chirp pattern distributions
     * Statistical measures of timing and frequency

4. Research Features:
   - Supports experimental designs with multiple subjects and conditions
   - Tracks chirp pattern transitions and sequences
   - Enables quantitative analysis of cricket acoustic behavior

Primary use case: Analyzing cricket communication patterns in research settings, with particular focus
on syllable timing, chirp structure, and pattern sequences in cricket calls.
"""