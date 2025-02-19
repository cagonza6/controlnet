import logging

import tempfile
from pathlib import Path

import imageio
from fastapi import APIRouter
from fastapi import Depends, UploadFile, File
from starlette.background import BackgroundTasks
from starlette.responses import StreamingResponse

from .ml import build_names, compress, generate_images, cleanup_file, model
from ..schemas.base import GenerationRequest

base_router = APIRouter()
logger = logging.getLogger("myawesomedemo")


@base_router.post("/generate")
async def generate(background_tasks: BackgroundTasks,
                   base_params: GenerationRequest = Depends(),
                   image: UploadFile = File(...)
                   ) -> StreamingResponse:
    """General end poit to: call the model and generate the files
    Deliver the files

    :param background_tasks: action to remove temporary files
    :param base_params: parameters to pass directly to the process generation
        the images
    :param image: Original image
    :return:
    """
    logger.info("Request started")
    params = base_params.dict()
    filename = Path(image.filename)
    extension = filename.suffix

    logger.info("Expected [{}] images to be generated.".format(params['num_samples']))

    if extension.lower() not in [".jpg", ".jpeg", ".png"]:
        raise ValueError("Unexpected extension for the provide image. Expected"
                         "'jpg', 'jpeg' or 'png'. obtained: {}".format(extension))

    gen_names = build_names(params['num_samples'], extension)

    # read image as numpy
    logger.info("Reading image.")
    image = imageio.imread(image.file.read())
    logger.info("Starting generation")
    gen_images = await generate_images(model, image, params)

    # This is a hack: create temporal file with zip extension and let it
    # be deleted, here I have a unique name. Reuse the name for the temporal
    # compressed file. Be aware of memory constrains....
    tf = tempfile.NamedTemporaryFile(suffix='.zip')
    tf.close()

    temp_zip_path = tf.name
    logger.info("Starting compression")
    await compress(gen_images, gen_names, params, temp_zip_path)

    def iter_file(some_file_path):
        """Helper to return the files as chunks"""
        with open(some_file_path, mode="rb") as file_like:
            yield from file_like

    download_name = filename.stem + '.zip'

    logger.info("Request: ready to respond")
    return StreamingResponse(
        iter_file(temp_zip_path),
        background=background_tasks.add_task(cleanup_file,
                                             Path(temp_zip_path)
                                             ),
        media_type="application/x-zip-compressed",
        headers={'Content-Disposition':
                     f'attachment; filename="{download_name}"'}
    )
