#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import logging


logging.basicConfig(level=logging.WARNING)


class PvlistParser:
    def __init__(self) -> None:
        self.pvlists: list[dict | None] = []

    def read_file(self, filepath):
        logging.info("open file {}".format(filepath))
        filename = os.path.basename(filepath)
        section = filename.replace("epicsArch_", "").replace(".txt", "").upper()
        pvlist = []
        with open(filepath, 'r') as main_file:
            for line_number, line in enumerate(main_file.readlines(), start=1):
                line = line.replace("\n", "").replace("\r", "")
                logging.debug(f"line: {line}")

                if not line:
                    logging.debug(f"{filename} line #{line_number}: empty line")
                    continue

                if line.startswith("#"):
                    logging.debug(f"{filename} line #{line_number}: comment line")
                elif line.startswith("*"):
                    logging.debug(f"{filename} line #{line_number}: description line")
                elif line.startswith("<"):
                    logging.debug("get_extrnal_file")
                    self.get_external_file(line)
                else:
                    try:
                        pvname, protocol = line.split(" ")
                    except ValueError:
                        pvname = line
                    pvlist.append({"name": pvname})
        if pvlist:
            self.pvlists.append({"SECTION": section, "PARAMS": pvlist})

    def get_external_file(self, line):
        basepath = '/cds/home/opr/tmoopr/misc/'
        _, filename = line.split(" ")
        filepath = "{}/{}".format(basepath, filename)
        self.read_file(filepath)


if __name__ == '__main__':
    # basepath = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(prog='Elog PVlist parser')
    parser.add_argument('-f', help="main file path", type=str, dest="filepath")

    args = parser.parse_args()
    main_file = args.filepath

    assert (os.path.isfile(main_file)), "file not found"
    pvlist_parser = PvlistParser()
    pvlist_parser.read_file(main_file)
    print(pvlist_parser.pvlists)
