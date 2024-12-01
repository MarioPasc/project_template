#!/usr/bin/env python3

import datetime
import platform
import getpass
import time

N: int = 91

def print_code_info():
    """
    Prints information about the code, authors, execution date, and machine.
    """
    print("=" * N)
    print("Welcome to the code execution of the paper: ")
    print("'Explorando la Demencia Frontotemporal mediante Biología de Sistemas: Un Enfoque Integrado'")
    print("-" * N)
    print("Code Version: 1.0")
    print("Authors: ")
    print("     - Mario Pascual González")
    print("     - Ainhoa Nerea Santana Bastante")
    print("     - Carmen Rodríguez González")
    print("     - Gonzalo Mesas Aranda")
    print("     - Ainhoa Pérez González")
    print("University of Málaga; Bioinformatics Minor; Systems Biology Course, 2024-2025")
    print(f"Execution Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Executing User: {getpass.getuser()}")
    print(f"Machine: {platform.node()}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    print("=" * N)
    print()

def print_free_use_statement():
    """
    Prints a statement about free use and authorship of the code and research.
    """
    print("=" * N)
    print("Free Use Statement")
    print("-" * N)
    print("This code is made available for free use under the condition that proper")
    print("credit is given to the authors. The research and development efforts")
    print("behind this work were conducted solely by the authors")
    print("Unauthorized claims of authorship are strictly prohibited.")
    print()
    print("We would like to thank all the professors involved in the Systems Biology")
    print("course, especially:")
    print("- Dr. James Richard Perkins for his outstanding mentorship")
    print("  throughout the entire project as our tutor.")
    print("- Dr. Pedro Seoane Zonjic for his guidance during the")
    print("  development of this code.")
    print("=" * N)
    print()

def main() -> None:
    print_code_info()
    time.sleep(3)
    print_free_use_statement()

if __name__ == "__main__":
    main()
