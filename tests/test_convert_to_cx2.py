import pytest
import sys
from os.path import abspath, dirname
from ndex2.cx2 import CX2Network

# Add the python_scripts directory to the sys.path
sys.path.insert(0, abspath(dirname(__file__) + '/../python_scripts'))

from convert_to_cx2 import *

def test_add_style_to_network_no_style():
    net = CX2Network()
    add_style_to_network(cx2_network=net)
    assert net.get_visual_properties() == []

def test_add_style_to_network_withinternal_style():
    net = CX2Network()

    style_cx_path = os.path.join(abspath(dirname(__file__)), '..',
                                         'data','style.cx2')
    assert net.get_visual_properties() == []
    add_style_to_network(cx2_network=net,style_path=style_cx_path)
    v_props = net.get_visual_properties()
    assert isinstance(v_props, dict)
    assert len(v_props.keys()) > 0

if __name__ == "__main__":
    pytest.main([__file__])
