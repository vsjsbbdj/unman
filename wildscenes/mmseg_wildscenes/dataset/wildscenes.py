from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset
import os.path as osp
import csv

def _get_reversed_custom_label_map(custom_label_map):
        rev_custom_label_map = {}
        label_set = set()
        for k,v in custom_label_map.items():
            if v not in label_set:
                rev_custom_label_map[v] = [k]
            else:
                rev_custom_label_map[v].append(k)
            label_set.add(v)
        
        return rev_custom_label_map


@DATASETS.register_module()
class WildscenesDataset(BaseSegDataset):
    """The Wildscenes datasets.

    Can be pointed to a specific split by setting data_root to a specific directory."""

    METAINFO = {
        "classes": (
            "unlabelled",
            "asphalt/concrete",
            "dirt",
            "mud",
            "water",
            "gravel",
            "other-terrain",
            "tree-trunk",
            "tree-foliage",
            "bush",
            "fence",
            "other-structure",
            "pole",
            "vehicle",
            "rock",
            "log",
            "other-object",
            "sky",
            "grass",
        ),
        "palette": [
          (0, 0, 0),
          (255, 165, 0),
          (60, 180, 75),
          (255, 225, 25),
          (0, 130, 200),
          (145, 30, 180),
          (70, 240, 240),
          (240, 50, 230),
          (210, 245, 60),
          (230, 25, 75),
          (0, 128, 128),
          (170, 110, 40),
          (255, 250, 200),
          (128, 0, 0),
          (170, 255, 195),
          (128, 128, 0),
          (250, 190, 190),
          (0, 0, 128),
          (128, 128, 128),
        ],
    }

    def __init__(
        self,
        img_suffix=".png",
        seg_map_suffix=".png",
        custom_label_map=None,
        ignore_index=255,
        reduce_zero_label=False,
        **kwargs,
    ):
        if reduce_zero_label:
            raise ValueError(
                "reduce_zero_label cannot be false. We ignore it in the label map"
            )
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            lazy_init=True,
            **kwargs,
        )
        # Apply our custom label map
        if custom_label_map is None:
            return
        self._check_custom_label_map(custom_label_map)
        idx_custom_label_map = self._get_idx_map(custom_label_map)
        new_classes = self._get_new_labels(custom_label_map)
        new_palette = self._get_updated_palette(custom_label_map)
        self.label_map = idx_custom_label_map
        new_metainfo = dict(
            palette=new_palette,
            classes=new_classes,
            label_map=idx_custom_label_map,
            reduce_zero_label=False,
        )
        self._metainfo = new_metainfo
        self._fully_initialized = False
        # NOTE: If we just completely overwrite the basesegdataset init then we won't run this twice
        self.full_init()
        print(f'Metainfo is {self._metainfo}.')

    def load_data_list(self):
        """Load annotations from CSV.

        支持两种格式：
        1) 带表头：id,im_path,label_path（im_path/label_path 为相对 data_root 或绝对路径）
        2) 无表头：每行一个 id（不带扩展名），将用 data_prefix 和后缀拼接
        """
        import csv
        import os.path as osp

        data_list = []
        ann_files = self.ann_file
        if not isinstance(ann_files, (list, tuple)):
            ann_files = [ann_files]

        for ann_file in ann_files:
            # ann_file 可为绝对路径或相对 data_root 的路径
            ann_path = ann_file if osp.isabs(ann_file) else osp.join(self.data_root, ann_file)

            with open(ann_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = [fn.strip() for fn in (reader.fieldnames or [])]

                if {'im_path', 'label_path'}.issubset(set(fieldnames)):
                    # CSV 含表头：直接用路径列
                    for row in reader:
                        im_rel = row['im_path'].strip()
                        seg_rel = row['label_path'].strip()
                        img_path = im_rel if osp.isabs(im_rel) else osp.join(self.data_root, im_rel)
                        seg_path = seg_rel if osp.isabs(seg_rel) else osp.join(self.data_root, seg_rel)
                        data_list.append(dict(
                            img_path=img_path,
                            seg_map_path=seg_path,
                            reduce_zero_label=getattr(self, 'reduce_zero_label', False),
                            seg_fields=[]
                        ))
                else:
                    # 无表头：每行一个 id（兼容旧格式）
                    f.seek(0)
                    for line in f:
                        _id = line.strip().split(',')[0]
                        if not _id:
                            continue
                        img_dir = self.data_prefix.get('img_path', '')
                        seg_dir = self.data_prefix.get('seg_map_path', '')
                        img_path = osp.join(self.data_root, img_dir, f'{_id}{self.img_suffix}')
                        seg_path = osp.join(self.data_root, seg_dir, f'{_id}{self.seg_map_suffix}')
                        data_list.append(dict(
                            img_path=img_path,
                            seg_map_path=seg_path,
                            reduce_zero_label=getattr(self, 'reduce_zero_label', False),
                            seg_fields=[]
                        ))

        return data_list
    def _get_idx_map(self, custom_label_map):
        """Get a idx to idx map from the label names in the custom label map"""
        # Ignore "unlabelled" - if its not present in the map it will be ignored automatically
        new_classes = self._get_new_labels(custom_label_map)
        custom_label_map = {
            k: v
            for k, v in custom_label_map.items()
            if k != "unlabelled" and v != "unlabelled"
        }
        # Sort alphabetically 
        orig_classes = self.METAINFO["classes"]
        idx_custom_label_map = {
            orig_classes.index(k): new_classes.index(v)
            for k, v in custom_label_map.items()
        }
        # Add missing classes
        idx_custom_label_map.update(
            {
                orig_classes.index(k): self.ignore_index
                for k in orig_classes
                if orig_classes.index(k) not in idx_custom_label_map
            }
        )
        return idx_custom_label_map

    def _check_custom_label_map(self, custom_label_map):
        """Check the label map for validity"""
        # Custom label map can only be set if custom metainfo is not set
        orig_classes = self.METAINFO["classes"]
        new_classes = list(set(custom_label_map.values()))
        if self.label_map is not None:
            raise ValueError(
                "Either custom_label_map or metainfo with new class labels can be set, but not both"
            )
        if len(new_classes) > len(orig_classes):
            raise ValueError(
                f"The class map has more classes ({len(new_classes)}) than the original dataset ({len(orig_classes)})"
            )
        if not all(old_cls in orig_classes for old_cls in custom_label_map.keys()):
            extra_classes = [
                old_cls
                for old_cls in custom_label_map.keys()
                if old_cls not in orig_classes
            ]
            raise ValueError(
                f"Map classes must map from original classes. {extra_classes} not in the original list"
            )
    
    def _get_new_labels(self, custom_label_map):
        """ Get the new labels from the label map"""
        new_classes = list(set(custom_label_map.values()))
        # Drop the unlabelled class (it's mapped to ignore_index)
        if "unlabelled" in new_classes:
            new_classes.remove("unlabelled")
        # Alphabetical order removes randomness from label ordering (different in different python processes)
        return sorted(new_classes)
    

    def _get_updated_palette(self, custom_label_map):
        """Update palette after applying the custom label map by assigning original palettes to existing classes 
        and remaining colours are assigned based on the label mapping from old to new classes"""

        rev_custom_label_map = _get_reversed_custom_label_map(custom_label_map)
        orig_palette = self.METAINFO["palette"]
        orig_classes = self.METAINFO["classes"]
        new_classes = self._get_new_labels(custom_label_map)
        remaining_idx = [
            i for i, cls in enumerate(orig_classes) if cls not in new_classes
        ]
        remaining_palette = [orig_palette[i] for i in remaining_idx]
        # Drop the first element because (0,0,0) shouldn't be assigned
        first_el = remaining_palette.pop(0)
        assert first_el == (0,0,0), "First element of the old palette should be (0,0,0)"
        new_palette = []
        for cls in new_classes:
            if cls in orig_classes:
                new_palette.append(orig_palette[orig_classes.index(cls)])
            else:
                new_palette.append(orig_palette[orig_classes.index(rev_custom_label_map[cls][-1])])
        return new_palette
