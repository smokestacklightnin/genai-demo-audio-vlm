from audiovlm_demo import AudioVLM, AudioVLMPanel


def main():
    A = AudioVLM()
    UI = AudioVLMPanel(engine=A)
    return UI


main()
