# Copyright 2020 IBM
# Author: peter.zhong@au1.ibm.com
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the Apache 2.0 License.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# Apache 2.0 License for more details.
#
# Vendored from OmniDocBench (Apache-2.0):
# https://github.com/opendatalab/OmniDocBench/blob/main/omnidocbench/metrics/table_metric.py
#
# Modifications:
# - Replaced `Levenshtein` package with `rapidfuzz.distance.Levenshtein`
# - Removed parallel processing, batch evaluation, tqdm dependency
# - Removed unused imports

from __future__ import annotations

from collections import deque

from apted import APTED, Config
from apted.helpers import Tree
from lxml import html
from rapidfuzz.distance import Levenshtein


class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == "td":
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % (
                self.tag,
                self.colspan,
                self.rowspan,
                self.content,
            )
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value"""
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1"""
        return float(Levenshtein.distance(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (
            (node1.tag != node2.tag)
            or (node1.colspan != node2.colspan)
            or (node1.rowspan != node2.rowspan)
        ):
            return 1.0
        if node1.tag == "td":
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.0


class TEDS:
    """Tree Edit Distance based Similarity"""

    def __init__(self, structure_only: bool = False, ignore_nodes=None):
        self.structure_only = structure_only
        self.ignore_nodes = ignore_nodes
        self.__tokens__: list[str] = []

    def tokenize(self, node):
        """Tokenizes table cells"""
        self.__tokens__.append("<%s>" % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != "unk":
            self.__tokens__.append("</%s>" % node.tag)
        if node.tag != "td" and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        """Converts HTML tree to the format required by apted"""
        if node.tag == "td":
            if self.structure_only:
                cell: list[str] = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(
                node.tag,
                int(node.attrib.get("colspan", "1")),
                int(node.attrib.get("rowspan", "1")),
                cell,
                *deque(),
            )
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred: str, true: str) -> float:
        """Computes TEDS score between prediction and ground truth HTML tables.

        Args:
            pred: Predicted HTML table string
            true: Ground truth HTML table string

        Returns:
            TEDS score between 0.0 and 1.0
        """
        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        pred_tree = html.fromstring(pred, parser=parser)
        true_tree = html.fromstring(true, parser=parser)
        if pred_tree.xpath("body/table") and true_tree.xpath("body/table"):
            pred_table = pred_tree.xpath("body/table")[0]
            true_table = true_tree.xpath("body/table")[0]
            if self.ignore_nodes:
                from lxml import etree

                etree.strip_tags(pred_table, *self.ignore_nodes)
                etree.strip_tags(true_table, *self.ignore_nodes)
            n_nodes_pred = len(pred_table.xpath(".//*"))
            n_nodes_true = len(true_table.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred_table)
            tree_true = self.load_html_tree(true_table)
            distance = APTED(
                tree_pred, tree_true, CustomConfig()
            ).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0
