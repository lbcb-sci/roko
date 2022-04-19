venv: roko_venv/bin/activate

roko_venv/bin/activate:
	python3 -m venv roko_venv
	. roko_venv/bin/activate; pip install pip --upgrade

libhts.a: Dependencies/htslib-1.9
	cd Dependencies/htslib-1.9; chmod +x ./configure ./version.sh
	. roko_venv/bin/activate; cd Dependencies/htslib-1.9; ./configure CFLAGS=-fpic --disable-bz2 --disable-lzma  --without-libdeflate && make

gpu: venv requirements.txt libhts.a generate_features.cpp models.cpp gen.cpp setup.py
	. roko_venv/bin/activate; pip install -r requirements.txt;
	. roko_venv/bin/activate; python3 setup.py build_ext; python3 setup.py install
	rm -rf build

cpu: venv requirements_cpu.txt libhts.a generate_features.cpp models.cpp gen.cpp setup.py
	. roko_venv/bin/activate; pip install -r requirements_cpu.txt; pip install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
	. roko_venv/bin/activate; python3 setup.py build_ext; python3 setup.py install
	rm -rf build

clean:
	cd Dependencies/htslib-1.9 && make clean || exit 0
	rm -rf roko_venv

