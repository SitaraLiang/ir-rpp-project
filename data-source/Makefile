all:
	wget https://lenskit.blob.core.windows.net/tors-lexirecall/urls.txt
	make -C qrels
	make -C runs
clean:
	rm -f urls.txt
	make -C qrels clean
	make -C runs clean