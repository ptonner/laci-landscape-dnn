import cattr
from attr import fields
from collections import OrderedDict
from functools import reduce
from operator import or_

from typing import (  # noqa: F401, imported for Mypy.
    Callable,
    Dict,
    Mapping,
    Optional,
    Type,
)

import logging

logger = logging.getLogger(__name__)


def create_uniq_field_dis_func(*classes):
    # type: (*Type) -> Callable
    """Given attr classes, generate a disambiguation function.  The
    function is based on unique fields. This is adapdted from the main
    cattrs library to use *ALL* unique attributes of an attrs
    class. This is necessary to prevent issues with serializations
    missing the selected unique attribute (because it has a default value).

    """
    if len(classes) < 2:
        raise ValueError("At least two classes required.")
    cls_and_attrs = [(cl, set(at.name for at in fields(cl))) for cl in classes]
    if len([attrs for _, attrs in cls_and_attrs if len(attrs) == 0]) > 1:
        raise ValueError("At least two classes have no attributes.")
    # TODO: Deal with a single class having no required attrs.
    # For each class, attempt to generate a single unique required field.
    uniq_attrs_dict = OrderedDict()  # type: Dict[str, Type]
    cls_and_attrs.sort(key=lambda c_a: -len(c_a[1]))

    fallback = None  # If none match, try this.

    for i, (cl, cl_reqs) in enumerate(cls_and_attrs):
        other_classes = cls_and_attrs[i + 1:]
        if other_classes:
            other_reqs = reduce(or_, (c_a[1] for c_a in other_classes))
            uniq = cl_reqs - other_reqs
            if not uniq:
                m = "{} has no usable unique attributes.".format(cl)
                raise ValueError(m)

            for u in iter(uniq):
                uniq_attrs_dict[u] = cl
            # uniq_attrs_dict[next(iter(uniq))] = cl
        else:
            fallback = cl

    def dis_func(data):
        # type: (Mapping) -> Optional[Type]
        if not isinstance(data, Mapping):
            raise ValueError("Only input mappings are supported.")
        for k, v in uniq_attrs_dict.items():
            if k in data:
                return v
        return fallback

    return dis_func


class _Register_Meta(type):
    """The register metaclass, creates a new register container.
    """

    def __new__(meta, name, base, attrs, **kwargs):
        c = super().__new__(meta, name, base, attrs, **kwargs)
        c.register = []
        return c


class _Register(type, metaclass=_Register_Meta):
    """The register type, inherit this type and use as a metaclass for
    registered objects.

    """

    @classmethod
    def structure(cls, d, _):
        if len(cls.register) > 1:
            disambiguator = create_uniq_field_dis_func(*cls.register)

            c = disambiguator(d)
            try:
                return cattr.structure_attrs_fromdict(d, c)
            except TypeError as te:
                raise TypeError(
                    (
                        te.args[0]
                        + " with type {} and register {}".format(c, cls.register),
                    )
                )

        return cattr.structure_attrs_fromdict(d, cls.register[0])

    def __new__(meta, name, base, attrs, **kwargs):
        c = super().__new__(meta, name, base, attrs, **kwargs)
        meta.register.append(c)

        cattr.register_structure_hook(c, meta.structure)

        logger.debug("{} registered to {}: {}".format(
            name, meta, meta.register))
        return c